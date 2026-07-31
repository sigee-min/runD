#pragma once

#include <rund/task/handle/ref.hpp>
#include <rund/task/result.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace rund::task {
template <typename T> class TaskAwaiter;
} // namespace rund::task

namespace rund::detail::task {

template <typename T> [[nodiscard]] const void *ResultTag() noexcept {
  static constexpr std::byte tag{};
  return &tag;
}

template <typename T> class ResultHandle final {
  static_assert(std::is_copy_constructible_v<T>,
                "task results must be copy constructible");

public:
  ResultHandle() noexcept = default;
  ResultHandle(const ResultHandle &) = delete;
  ResultHandle &operator=(const ResultHandle &) = delete;

  ResultHandle(ResultHandle &&other) noexcept : ref_(other.take()) {}

  ResultHandle &operator=(ResultHandle &&other) noexcept {
    if (this != &other) {
      reset();
      ref_ = other.take();
    }
    return *this;
  }

  ~ResultHandle() { reset(); }

  [[nodiscard]] explicit operator bool() const noexcept {
    return ref_.authority != nullptr;
  }

  [[nodiscard]] ::rund::task::Poll poll() const noexcept {
    return ref_.poll != nullptr
               ? ref_.poll(ref_.authority, ref_.slot, ref_.generation)
               : ::rund::task::Poll{.phase = ::rund::task::Phase::Failed,
                                    .code = ReasonCode::TaskHandleStale};
  }

  [[nodiscard]] ::rund::task::Status wait() const noexcept {
    return ref_.wait != nullptr
               ? ref_.wait(ref_.authority, ref_.slot, ref_.generation)
               : ::rund::task::Status::fail(ReasonCode::TaskHandleStale);
  }

  [[nodiscard]] ::rund::task::Result<T> result() const {
    if (ref_.copy == nullptr) {
      return ::rund::task::Result<T>::fail(ReasonCode::TaskHandleStale);
    }
    alignas(T) std::byte storage[sizeof(T)];
    const ::rund::task::Status copied =
        ref_.copy(ref_.authority, ref_.slot, ref_.generation, storage,
                  ResultTag<T>(), &Copy);
    if (!copied) {
      return ::rund::task::Result<T>::fail(copied.code());
    }
    T *const value = reinterpret_cast<T *>(storage);
    ::rund::task::Result<T> out =
        ::rund::task::Result<T>::success(std::move(*value));
    value->~T();
    return out;
  }

  explicit ResultHandle(ResultRef ref) noexcept : ref_(ref) {}

private:
  friend class ::rund::task::TaskAwaiter<T>;

  static void Copy(void *out, const void *value) {
    new (out) T(*static_cast<const T *>(value));
  }

  [[nodiscard]] ResultRef take() noexcept {
    const ResultRef ref = ref_;
    ref_ = {};
    return ref;
  }

  void reset() noexcept {
    if (ref_.release != nullptr) {
      ref_.release(ref_.authority, ref_.slot, ref_.generation);
    }
    ref_ = {};
  }

  ResultRef ref_{};
};

template <> class ResultHandle<void> final {
public:
  ResultHandle() noexcept = default;
  ResultHandle(const ResultHandle &) = delete;
  ResultHandle &operator=(const ResultHandle &) = delete;

  ResultHandle(ResultHandle &&other) noexcept : ref_(other.take()) {}

  ResultHandle &operator=(ResultHandle &&other) noexcept {
    if (this != &other) {
      reset();
      ref_ = other.take();
    }
    return *this;
  }

  ~ResultHandle() { reset(); }

  [[nodiscard]] explicit operator bool() const noexcept {
    return ref_.authority != nullptr;
  }

  [[nodiscard]] ::rund::task::Poll poll() const noexcept {
    return ref_.poll != nullptr
               ? ref_.poll(ref_.authority, ref_.slot, ref_.generation)
               : ::rund::task::Poll{.phase = ::rund::task::Phase::Failed,
                                    .code = ReasonCode::TaskHandleStale};
  }

  [[nodiscard]] ::rund::task::Status wait() const noexcept {
    return ref_.wait != nullptr
               ? ref_.wait(ref_.authority, ref_.slot, ref_.generation)
               : ::rund::task::Status::fail(ReasonCode::TaskHandleStale);
  }

  [[nodiscard]] ::rund::task::Result<void> result() const noexcept {
    const ::rund::task::Poll state = poll();
    if (!state.terminal()) {
      return ::rund::task::Result<void>::fail(ReasonCode::TaskFailed);
    }
    return state.code == ReasonCode::Ok
               ? ::rund::task::Result<void>::success()
               : ::rund::task::Result<void>::fail(state.code);
  }

  explicit ResultHandle(ResultRef ref) noexcept : ref_(ref) {}

private:
  friend class ::rund::task::TaskAwaiter<void>;

  [[nodiscard]] ResultRef take() noexcept {
    const ResultRef ref = ref_;
    ref_ = {};
    return ref;
  }

  void reset() noexcept {
    if (ref_.release != nullptr) {
      ref_.release(ref_.authority, ref_.slot, ref_.generation);
    }
    ref_ = {};
  }

  ResultRef ref_{};
};

} // namespace rund::detail::task
