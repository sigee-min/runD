#pragma once

#include <rund/net/ready/ticket.hpp>
#include <rund/task/await.hpp>

#include <coroutine>
#include <utility>

namespace rund::net::ready {

class Awaiter;

class Wait final {
public:
  Wait(const Wait &) = delete;
  Wait &operator=(const Wait &) = delete;
  Wait(Wait &&) noexcept = default;
  Wait &operator=(Wait &&) noexcept = default;

  [[nodiscard]] Ticket wait() && noexcept;

private:
  friend class Awaiter;
  friend struct detail::Access;

  Wait() noexcept = default;

  task::IoOp operation_ = ::rund::detail::task::OpAccess::io(
      task::Status::fail(::rund::ReasonCode::TaskInvalid), 0, false, -1, 0, 0u,
      0u);
  SocketView socket_{};
  Interest interest_ = Interest::Readable;
  short native_interest_ = 0;
  bool deferred_ = false;
};

class Awaiter final {
public:
  explicit Awaiter(Wait operation) noexcept
      : socket_(operation.socket_), interest_(operation.interest_),
        native_interest_(operation.native_interest_),
        deferred_(operation.deferred_), operation_(operation.operation_) {}

  [[nodiscard]] bool await_ready() const noexcept {
    return operation_.await_ready();
  }
  bool await_suspend(std::coroutine_handle<> handle) noexcept;
  [[nodiscard]] Ticket await_resume() noexcept;

private:
  SocketView socket_{};
  Interest interest_ = Interest::Readable;
  short native_interest_ = 0;
  bool deferred_ = false;
  task::IoAwaiter operation_;
  ::rund::detail::task::IoDecision decision_{};
};

[[nodiscard]] Wait read(SocketView socket) noexcept;
[[nodiscard]] Wait write(SocketView socket) noexcept;

[[nodiscard]] inline Awaiter operator co_await(Wait operation) noexcept {
  return Awaiter{std::move(operation)};
}

} // namespace rund::net::ready
