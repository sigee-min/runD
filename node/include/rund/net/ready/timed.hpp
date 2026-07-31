#pragma once

#include <rund/net/ready/ticket.hpp>

#include <chrono>
#include <coroutine>
#include <cstdint>
#include <utility>

namespace rund::detail::task {
class AwaitAccess;
}

namespace rund::net::ready::timed {

namespace detail {
class Access;
}

class Awaiter;

class Wait final {
public:
  Wait() noexcept = default;
  Wait(const Wait &) = delete;
  Wait &operator=(const Wait &) = delete;
  Wait(Wait &&other) noexcept { move_from(other); }
  Wait &operator=(Wait &&other) noexcept {
    if (this != &other) {
      move_from(other);
    }
    return *this;
  }

  [[nodiscard]] Ticket wait() && noexcept;

private:
  friend class Awaiter;
  friend class detail::Access;
  friend class ::rund::detail::task::AwaitAccess;

  void move_from(Wait &other) noexcept {
    result_ = std::move(other.result_);
    deferred_ = std::exchange(other.deferred_, false);
    suspended_ = std::exchange(other.suspended_, false);
    task_id_ = std::exchange(other.task_id_, 0u);
    socket_ = std::exchange(other.socket_, SocketView{});
    public_interest_ =
        std::exchange(other.public_interest_, Interest::Readable);
    interest_ = std::exchange(other.interest_, 0);
    timeout_ns_ = std::exchange(other.timeout_ns_, 0);
    stop_scheduler_id_ = std::exchange(other.stop_scheduler_id_, 0u);
    stop_source_id_ = std::exchange(other.stop_source_id_, 0u);
    stop_generation_ = std::exchange(other.stop_generation_, 0u);
    stop_epoch_ = std::exchange(other.stop_epoch_, 0u);
  }

  Ticket result_{};
  bool deferred_ = false;
  bool suspended_ = false;
  std::uint64_t task_id_ = 0u;
  SocketView socket_{};
  Interest public_interest_ = Interest::Readable;
  short interest_ = 0;
  std::int64_t timeout_ns_ = 0;
  std::uint64_t stop_scheduler_id_ = 0u;
  std::uint64_t stop_source_id_ = 0u;
  std::uint64_t stop_generation_ = 0u;
  std::uint64_t stop_epoch_ = 0u;
};

class Awaiter final {
public:
  explicit Awaiter(Wait operation) noexcept
      : operation_(std::move(operation)) {}

  [[nodiscard]] bool await_ready() const noexcept {
    return !operation_.deferred_ && !operation_.suspended_;
  }

  bool await_suspend(std::coroutine_handle<>) noexcept;

  [[nodiscard]] Ticket await_resume() noexcept;

private:
  Wait operation_{};
};

[[nodiscard]] Wait read(SocketView socket,
                        std::chrono::nanoseconds timeout) noexcept;
[[nodiscard]] Wait write(SocketView socket,
                         std::chrono::nanoseconds timeout) noexcept;

[[nodiscard]] inline Awaiter operator co_await(Wait operation) noexcept {
  return Awaiter{std::move(operation)};
}

} // namespace rund::net::ready::timed
