#pragma once

#include <rund/task/handle/typed.hpp>

#include <coroutine>
#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rund::task {

template <typename T> class Task;

} // namespace rund::task

namespace rund::detail::task {

struct CoroutineResult final {
  void *value{};
  std::size_t bytes{};
  std::size_t alignment{};
  const void *type{};
  void (*move)(void *, void *){};
  void (*destroy)(void *) noexcept {};
  bool has_value{};
  ReasonCode code{ReasonCode::Ok};
};

struct CoroutineOps final {
  CoroutineResult (*result)(std::coroutine_handle<> frame){};
  void (*destroy)(std::coroutine_handle<> frame) noexcept {};
};

struct CoroutineStart final {
  std::coroutine_handle<> frame{};
  const CoroutineOps *ops{};
  ReasonCode code{ReasonCode::Ok};
};

namespace frame {
[[nodiscard]] void *Acquire(std::size_t bytes, std::size_t alignment) noexcept;
void Release(void *frame) noexcept;
[[nodiscard]] ReasonCode TakeFailure() noexcept;
[[nodiscard]] std::uint32_t Bytes(void *frame) noexcept;
[[nodiscard]] bool Reused(void *frame) noexcept;
} // namespace frame

class alignas(16) PromiseFrame {
public:
  [[nodiscard]] static void *operator new(const std::size_t bytes) noexcept {
    return frame::Acquire(bytes, alignof(PromiseFrame));
  }

  static void operator delete(void *value, std::size_t) noexcept {
    frame::Release(value);
  }
};

struct FinalSuspend final {
  [[nodiscard]] constexpr bool await_ready() const noexcept { return false; }

  template <typename Promise>
  constexpr void await_suspend(std::coroutine_handle<Promise>) const noexcept {}

  constexpr void await_resume() const noexcept {}
};

template <typename T> class Promise : public PromiseFrame {
public:
  using value_type = T;

  [[nodiscard]] ::rund::task::Task<T> get_return_object() noexcept;
  [[nodiscard]] static ::rund::task::Task<T>
  get_return_object_on_allocation_failure() noexcept;
  [[nodiscard]] constexpr std::suspend_always initial_suspend() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr FinalSuspend final_suspend() const noexcept {
    return {};
  }

  template <typename U>
    requires std::is_constructible_v<T, U &&>
  void
  return_value(U &&value) noexcept(std::is_nothrow_constructible_v<T, U &&>) {
    value_.emplace(std::forward<U>(value));
  }

  void unhandled_exception() noexcept { failure_ = ReasonCode::TaskFailed; }

  [[nodiscard]] ReasonCode failure() const noexcept { return failure_; }
  [[nodiscard]] T *value() noexcept {
    return value_ ? std::addressof(*value_) : nullptr;
  }

private:
  ReasonCode failure_{ReasonCode::Ok};
  std::optional<T> value_{};
};

template <> class Promise<void> : public PromiseFrame {
public:
  using value_type = void;

  [[nodiscard]] ::rund::task::Task<void> get_return_object() noexcept;
  [[nodiscard]] static ::rund::task::Task<void>
  get_return_object_on_allocation_failure() noexcept;
  [[nodiscard]] constexpr std::suspend_always initial_suspend() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr FinalSuspend final_suspend() const noexcept {
    return {};
  }

  constexpr void return_void() const noexcept {}

  void unhandled_exception() noexcept { failure_ = ReasonCode::TaskFailed; }

  [[nodiscard]] ReasonCode failure() const noexcept { return failure_; }

private:
  ReasonCode failure_{ReasonCode::Ok};
};

template <typename T>
[[nodiscard]] CoroutineStart
TakeCoroutine(::rund::task::Task<T> &&task) noexcept;

template <typename T>
[[nodiscard]] void *
CoroutineAddress(const ::rund::task::Task<T> &task) noexcept;

} // namespace rund::detail::task

namespace rund::task {

template <typename T> class Task {
public:
  using promise_type = ::rund::detail::task::Promise<T>;

  constexpr Task() noexcept = default;
  explicit constexpr Task(const ReasonCode code) noexcept : code_(code) {}

  Task(const Task &) = delete;
  Task &operator=(const Task &) = delete;

  constexpr Task(Task &&other) noexcept
      : frame_(other.release()), code_(other.code_) {}

  Task &operator=(Task &&other) noexcept {
    if (this != &other) {
      destroy();
      frame_ = other.release();
      code_ = other.code_;
    }
    return *this;
  }

  ~Task() { destroy(); }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return frame_ != nullptr;
  }

  [[nodiscard]] constexpr ReasonCode code() const noexcept {
    return frame_ == nullptr && code_ == ReasonCode::Ok
               ? ReasonCode::TaskInvalid
               : code_;
  }

  [[nodiscard]] std::string_view error() const noexcept {
    return code() == ReasonCode::Ok ? std::string_view{}
                                    : std::string_view{ReasonString(code())};
  }

  [[nodiscard]] bool done() const noexcept {
    return frame_ == nullptr || frame_.done();
  }

private:
  using handle_type = std::coroutine_handle<promise_type>;

  template <typename U>
  friend auto ::rund::detail::task::TakeCoroutine(Task<U> &&task) noexcept
      -> ::rund::detail::task::CoroutineStart;
  template <typename U>
  friend void * ::rund::detail::task::CoroutineAddress(
      const Task<U> &task) noexcept;
  friend class ::rund::detail::task::Promise<T>;

  explicit constexpr Task(handle_type frame) noexcept : frame_(frame) {}

  [[nodiscard]] constexpr handle_type release() noexcept {
    handle_type frame = frame_;
    frame_ = nullptr;
    return frame;
  }

  void destroy() noexcept {
    if (frame_ != nullptr) {
      frame_.destroy();
      frame_ = nullptr;
    }
  }

  handle_type frame_{};
  ReasonCode code_{ReasonCode::Ok};
};

} // namespace rund::task

namespace rund::detail::task {

template <typename T>
void MoveCoroutineResult(void *const out, void *const value) {
  new (out) T(std::move(*static_cast<T *>(value)));
}

template <typename T> void DestroyCoroutineResult(void *const value) noexcept {
  static_cast<T *>(value)->~T();
}

template <typename T>
CoroutineResult ReadCoroutineResult(const std::coroutine_handle<> frame) {
  auto typed = std::coroutine_handle<Promise<T>>::from_address(frame.address());
  Promise<T> &promise = typed.promise();
  T *const value = promise.value();
  if (promise.failure() != ReasonCode::Ok || value == nullptr) {
    return CoroutineResult{.code = promise.failure() == ReasonCode::Ok
                                       ? ReasonCode::TaskFailed
                                       : promise.failure()};
  }
  return CoroutineResult{.value = value,
                         .bytes = sizeof(T),
                         .alignment = alignof(T),
                         .type = ResultTag<T>(),
                         .move = &MoveCoroutineResult<T>,
                         .destroy = &DestroyCoroutineResult<T>,
                         .has_value = true,
                         .code = ReasonCode::Ok};
}

inline CoroutineResult
ReadVoidCoroutineResult(const std::coroutine_handle<> frame) {
  auto typed =
      std::coroutine_handle<Promise<void>>::from_address(frame.address());
  return CoroutineResult{.code = typed.promise().failure()};
}

template <typename T>
void DestroyCoroutine(const std::coroutine_handle<> frame) noexcept {
  std::coroutine_handle<Promise<T>>::from_address(frame.address()).destroy();
}

inline void DestroyCoroutine(CoroutineStart &start) noexcept {
  const std::coroutine_handle<> frame = std::exchange(start.frame, {});
  const CoroutineOps *const ops = std::exchange(start.ops, nullptr);
  start.code = ReasonCode::TaskInvalid;
  if (frame != nullptr && ops != nullptr && ops->destroy != nullptr) {
    ops->destroy(frame);
  }
}

template <typename T>
CoroutineStart TakeCoroutine(::rund::task::Task<T> &&task) noexcept {
  static constexpr CoroutineOps ops = [] {
    if constexpr (std::is_void_v<T>) {
      return CoroutineOps{.result = &ReadVoidCoroutineResult,
                          .destroy = &DestroyCoroutine<T>};
    } else {
      return CoroutineOps{.result = &ReadCoroutineResult<T>,
                          .destroy = &DestroyCoroutine<T>};
    }
  }();
  const ReasonCode code = task.code();
  const auto typed = task.release();
  return CoroutineStart{.frame = typed, .ops = &ops, .code = code};
}

template <typename T>
void *CoroutineAddress(const ::rund::task::Task<T> &task) noexcept {
  return task.frame_ == nullptr ? nullptr : task.frame_.address();
}

template <typename T>
::rund::task::Task<T> Promise<T>::get_return_object() noexcept {
  return ::rund::task::Task<T>{
      std::coroutine_handle<Promise<T>>::from_promise(*this)};
}

template <typename T>
::rund::task::Task<T>
Promise<T>::get_return_object_on_allocation_failure() noexcept {
  return ::rund::task::Task<T>{frame::TakeFailure()};
}

inline ::rund::task::Task<void> Promise<void>::get_return_object() noexcept {
  return ::rund::task::Task<void>{
      std::coroutine_handle<Promise<void>>::from_promise(*this)};
}

inline ::rund::task::Task<void>
Promise<void>::get_return_object_on_allocation_failure() noexcept {
  return ::rund::task::Task<void>{frame::TakeFailure()};
}

} // namespace rund::detail::task
