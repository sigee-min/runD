#pragma once

#include <rund/net/ready/ticket.hpp>

#include <chrono>
#include <coroutine>
#include <cstdint>
#include <span>
#include <utility>

namespace rund::net::ready {

struct Request {
  SocketView socket{};
  Interest interest = Interest::Readable;
};

struct Event {
  std::uint32_t index = 0u;
  Ticket ticket{};
};

struct Set;

namespace many {

struct Budget {
  std::uint32_t max_events = 64u;
};

struct Result : net::Status {
  using net::Status::Status;

  std::uint32_t events = 0u;
  bool budget_exhausted = false;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return ::rund::detail::timed_status_ok(code());
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::detail::timed_status_error(code());
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ::rund::detail::timed_status_exit(code());
  }
  [[nodiscard]] constexpr bool timed_out() const noexcept {
    return code() == ::rund::ReasonCode::IoTimedOut;
  }
};

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

  [[nodiscard]] Result wait() const noexcept;

private:
  friend class Awaiter;
  friend class detail::Access;

  void move_from(Wait &other) noexcept {
    result_ = std::exchange(other.result_, Result{});
    deferred_ = std::exchange(other.deferred_, false);
    suspended_ = std::exchange(other.suspended_, false);
    requests_ = std::exchange(other.requests_, {});
    out_ = std::exchange(other.out_, {});
    budget_ = std::exchange(other.budget_, Budget{});
    timeout_ns_ = std::exchange(other.timeout_ns_, 0);
    has_timeout_ = std::exchange(other.has_timeout_, false);
    group_id_ = std::exchange(other.group_id_, 0u);
    ready_set_id_ = std::exchange(other.ready_set_id_, 0u);
    ready_set_generation_ = std::exchange(other.ready_set_generation_, 0u);
    stop_scheduler_id_ = std::exchange(other.stop_scheduler_id_, 0u);
    stop_source_id_ = std::exchange(other.stop_source_id_, 0u);
    stop_generation_ = std::exchange(other.stop_generation_, 0u);
    stop_epoch_ = std::exchange(other.stop_epoch_, 0u);
  }

  Result result_{};
  bool deferred_ = false;
  bool suspended_ = false;
  std::span<const Request> requests_{};
  std::span<Event> out_{};
  Budget budget_{};
  std::int64_t timeout_ns_ = 0;
  bool has_timeout_ = false;
  std::uint64_t group_id_ = 0u;
  std::uint64_t ready_set_id_ = 0u;
  std::uint64_t ready_set_generation_ = 0u;
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
  [[nodiscard]] Result await_resume() noexcept;

private:
  Wait operation_{};
};

[[nodiscard]] Wait wait(std::span<const Request> requests, std::span<Event> out,
                        Budget budget = {}) noexcept;
[[nodiscard]] Wait wait(std::span<const Request> requests, std::span<Event> out,
                        std::chrono::nanoseconds timeout,
                        Budget budget = {}) noexcept;

[[nodiscard]] inline Awaiter operator co_await(Wait operation) noexcept {
  return Awaiter{std::move(operation)};
}

} // namespace many
} // namespace rund::net::ready
