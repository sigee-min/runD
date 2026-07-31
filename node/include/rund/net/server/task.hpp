#pragma once

#include <rund/net/server/result.hpp>
#include <rund/task/await.hpp>

#include <coroutine>
#include <utility>

namespace rund::net::server {

class Task final {
public:
  class Awaiter final {
  public:
    explicit Awaiter(task::Task<Result> task) noexcept
        : task_(std::move(task)) {}

    [[nodiscard]] bool await_ready() const noexcept {
      return task_.await_ready();
    }

    bool await_suspend(const std::coroutine_handle<> awaiting) noexcept {
      return task_.await_suspend(awaiting);
    }

    [[nodiscard]] Result await_resume() {
      task::Result<Result> result = task_.await_resume();
      return result ? *std::move(result) : Result{result.code()};
    }

  private:
    task::TaskAwaiter<Result> task_;
  };

  explicit Task(task::Task<Result> task) noexcept : task_(std::move(task)) {}

  Task(const Task &) = delete;
  Task &operator=(const Task &) = delete;
  Task(Task &&) noexcept = default;
  Task &operator=(Task &&) noexcept = default;

  [[nodiscard]] Awaiter operator co_await() && noexcept {
    return Awaiter{std::move(task_)};
  }

private:
  task::Task<Result> task_;
};

} // namespace rund::net::server
