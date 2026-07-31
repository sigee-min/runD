#pragma once

#include <rund/task/api/access.hpp>
#include <rund/task/await/access.hpp>
#include <rund/task/coroutine.hpp>
#include <rund/task/handle.hpp>
#include <rund/task/results.hpp>

#include <coroutine>
#include <utility>

namespace rund::task {

[[nodiscard]] Status join(Handle handle) noexcept;
[[nodiscard]] Handle spawn(const char *name, Task<void> &&task) noexcept;

class YieldAwaiter final {
public:
  explicit YieldAwaiter(const YieldOp op) noexcept : op_(op) {}

  [[nodiscard]] bool await_ready() const noexcept {
    return !op_.status_ || !op_.deferred_;
  }

  bool await_suspend(std::coroutine_handle<>) noexcept {
    decision_ = ::rund::detail::task::AwaitAccess::SuspendCoroutineYield();
    return decision_.status && decision_.suspend;
  }

  [[nodiscard]] Status await_resume() const noexcept {
    return op_.deferred_ ? decision_.status : op_.status_;
  }

private:
  YieldOp op_;
  ::rund::detail::task::AwaitDecision decision_{};
};

class SleepAwaiter final {
public:
  explicit SleepAwaiter(const SleepOp op) noexcept : op_(op) {}

  [[nodiscard]] bool await_ready() const noexcept {
    return !op_.status_ || !op_.deferred_;
  }

  bool await_suspend(std::coroutine_handle<>) noexcept {
    decision_ =
        ::rund::detail::task::AwaitAccess::SuspendCoroutineSleep(op_.duration_);
    return decision_.status && decision_.suspend;
  }

  [[nodiscard]] Status await_resume() const noexcept {
    return op_.deferred_ ? decision_.status : op_.status_;
  }

private:
  SleepOp op_;
  ::rund::detail::task::AwaitDecision decision_{};
};

class IoAwaiter final {
public:
  explicit IoAwaiter(const IoOp op) noexcept : op_(op) {}

  [[nodiscard]] bool await_ready() const noexcept {
    return !op_.result_ || !op_.deferred_;
  }

  bool await_suspend(std::coroutine_handle<>) noexcept {
    decision_ = ::rund::detail::task::AwaitAccess::SuspendCoroutineIo(
        op_.fd_, op_.interest_, op_.host_handle_id_, op_.fd_generation_);
    return decision_.status && decision_.suspend;
  }

  [[nodiscard]] IoResult await_resume() noexcept {
    if (!op_.deferred_) {
      return op_.result_;
    }
    return ::rund::detail::task::AwaitAccess::CompleteCoroutineIoAwait(
        decision_);
  }

private:
  IoOp op_;
  ::rund::detail::task::IoDecision decision_{};
};

class HandleAwaiter final {
public:
  explicit HandleAwaiter(const Handle &handle) noexcept : handle_(handle) {}

  [[nodiscard]] bool await_ready() const noexcept { return false; }

  bool await_suspend(std::coroutine_handle<>) noexcept {
    decision_ =
        ::rund::detail::task::AwaitAccess::BeginCoroutineJoinAwait(handle_);
    return decision_.status && decision_.suspend;
  }

  [[nodiscard]] Status await_resume() noexcept {
    const Status joined =
        ::rund::detail::task::AwaitAccess::CompleteCoroutineJoinAwait(
            decision_);
    ::rund::detail::task::AwaitAccess::RetireCoroutineTask(handle_);
    return joined;
  }

private:
  Handle handle_{};
  ::rund::detail::task::AwaitDecision decision_{};
};

template <typename T> class TaskAwaiter final {
  static constexpr bool kStatus = std::is_same_v<T, Status>;
  using AwaitResult = std::conditional_t<kStatus, Status, Result<T>>;

public:
  explicit TaskAwaiter(Task<T> &&task) noexcept : task_(std::move(task)) {}

  [[nodiscard]] bool await_ready() const noexcept { return false; }

  bool await_suspend(std::coroutine_handle<>) noexcept {
    ::rund::detail::task::Spawned spawned =
        ::rund::detail::task::ApiAccess::SpawnAwaited(
            ::rund::detail::task::TakeCoroutine(std::move(task_)));
    handle_ = spawned.task;
    result_handle_ = ::rund::detail::task::ResultHandle<T>{spawned.result};
    decision_ =
        ::rund::detail::task::AwaitAccess::BeginCoroutineJoinAwait(handle_);
    return decision_.status && decision_.suspend;
  }

  [[nodiscard]] AwaitResult await_resume() {
    const Status joined =
        ::rund::detail::task::AwaitAccess::CompleteCoroutineJoinAwait(
            decision_);
    if (!joined) {
      ::rund::detail::task::AwaitAccess::RetireCoroutineTask(handle_);
      if constexpr (kStatus) {
        return joined;
      } else {
        return Result<T>::fail(joined.code());
      }
    }
    Result<T> result = result_handle_.result();
    ::rund::detail::task::AwaitAccess::RetireCoroutineTask(handle_);
    if constexpr (kStatus) {
      return result ? *result : Status::fail(result.code());
    } else {
      return result;
    }
  }

private:
  Task<T> task_;
  Handle handle_{};
  ::rund::detail::task::ResultHandle<T> result_handle_{};
  ::rund::detail::task::AwaitDecision decision_{};
};

[[nodiscard]] inline YieldAwaiter operator co_await(YieldOp op) noexcept {
  return YieldAwaiter{op};
}

[[nodiscard]] inline SleepAwaiter operator co_await(SleepOp op) noexcept {
  return SleepAwaiter{op};
}

[[nodiscard]] inline IoAwaiter operator co_await(IoOp op) noexcept {
  return IoAwaiter{op};
}

[[nodiscard]] inline HandleAwaiter
operator co_await(const Handle &handle) noexcept {
  return HandleAwaiter{handle};
}

template <typename T>
[[nodiscard]] inline TaskAwaiter<T> operator co_await(Task<T> &&task) noexcept {
  return TaskAwaiter<T>{std::move(task)};
}

} // namespace rund::task
