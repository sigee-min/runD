#pragma once

#include <rund/task/await.hpp>

#include <rund/compute/session/request.hpp>
#include <rund/compute/session/submission.hpp>

#include <coroutine>
#include <memory>
#include <optional>
#include <utility>

namespace rund::compute {

class Request::Awaiter final {
public:
  Awaiter(std::weak_ptr<void> host, std::shared_ptr<void> operation,
          const void *operations) noexcept
      : host_(std::move(host)), operation_(std::move(operation)),
        operations_(operations) {}

  [[nodiscard]] constexpr bool await_ready() const noexcept { return false; }

  bool await_suspend(std::coroutine_handle<> continuation) noexcept {
    submission_ = Request{std::move(host_), std::move(operation_),
                          operations_}.submit();
    if (submission_.poll().completed) {
      return false;
    }
    join_.emplace(submission_.handle());
    return join_->await_suspend(continuation);
  }

  [[nodiscard]] Completion await_resume() noexcept {
    Completion completion = submission_.wait();
    if (join_.has_value()) {
      (void)join_->await_resume();
    }
    return completion;
  }

private:
  std::weak_ptr<void> host_{};
  std::shared_ptr<void> operation_{};
  const void *operations_ = nullptr;
  Submission submission_{};
  std::optional<task::HandleAwaiter> join_{};
};

inline Request::Awaiter Request::operator co_await() && noexcept {
  return Awaiter{std::move(host_), std::move(operation_), operations_};
}

} // namespace rund::compute
