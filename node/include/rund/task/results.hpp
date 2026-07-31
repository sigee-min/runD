#pragma once

#include <rund/host/chrono.hpp>
#include <rund/task/status.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rund::task {

class YieldOp;
class SleepOp;
class IoOp;
class Status;

} // namespace rund::task

namespace rund::detail::task {

struct AwaitDecision final {
  ::rund::task::Status status{};
  bool suspend = false;
};

struct IoDecision final {
  ::rund::task::Status status{};
  short revents = 0;
  bool suspend = false;
};

class OpAccess final {
public:
  OpAccess() = delete;

  [[nodiscard]] static ::rund::task::YieldOp yield(::rund::task::Status status,
                                                   bool deferred) noexcept;
  [[nodiscard]] static ::rund::task::SleepOp
  sleep(::rund::task::Status status, std::chrono::nanoseconds duration,
        bool deferred) noexcept;
  [[nodiscard]] static ::rund::task::IoOp
  io(::rund::task::Status status, short revents, bool deferred, int fd,
     short interest, std::uint64_t host_handle_id,
     std::uint64_t fd_generation) noexcept;
};

} // namespace rund::detail::task

namespace rund::task {

class YieldOp final {
public:
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return static_cast<bool>(status_);
  }
  [[nodiscard]] constexpr bool ok() const noexcept { return status_.ok(); }
  [[nodiscard]] constexpr ReasonCode code() const noexcept {
    return status_.code();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return status_.error();
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return status_.exit_code();
  }

private:
  friend class YieldAwaiter;
  friend class ::rund::detail::task::OpAccess;

  constexpr YieldOp(const Status status, const bool deferred) noexcept
      : status_(status), deferred_(deferred) {}

  Status status_{};
  bool deferred_ = false;
};

class SleepOp final {
public:
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return static_cast<bool>(status_);
  }
  [[nodiscard]] constexpr bool ok() const noexcept { return status_.ok(); }
  [[nodiscard]] constexpr ReasonCode code() const noexcept {
    return status_.code();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return status_.error();
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return status_.exit_code();
  }

private:
  friend class SleepAwaiter;
  friend class ::rund::detail::task::OpAccess;

  constexpr SleepOp(const Status status,
                    const std::chrono::nanoseconds duration,
                    const bool deferred) noexcept
      : duration_(duration), status_(status), deferred_(deferred) {}

  std::chrono::nanoseconds duration_{};
  Status status_{};
  bool deferred_ = false;
};

class IoResult final {
public:
  constexpr IoResult() noexcept = default;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return static_cast<bool>(status_);
  }
  [[nodiscard]] constexpr bool ok() const noexcept { return status_.ok(); }
  [[nodiscard]] constexpr ReasonCode code() const noexcept {
    return status_.code();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return status_.error();
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return status_.exit_code();
  }
  [[nodiscard]] constexpr short revents() const noexcept { return revents_; }

  [[nodiscard]] static constexpr IoResult
  success(const short revents = 0) noexcept {
    return IoResult{Status::success(), revents};
  }

  [[nodiscard]] static constexpr IoResult
  fail(const ReasonCode code, const short revents = 0) noexcept {
    return IoResult{Status::fail(code), revents};
  }

private:
  friend class IoAwaiter;
  friend class ::rund::detail::task::OpAccess;

  constexpr IoResult(const Status status, const short revents) noexcept
      : status_(status), revents_(revents) {}

  Status status_{};
  short revents_ = 0;
};

class IoOp final {
public:
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return static_cast<bool>(result_);
  }
  [[nodiscard]] constexpr bool ok() const noexcept { return result_.ok(); }
  [[nodiscard]] constexpr ReasonCode code() const noexcept {
    return result_.code();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return result_.error();
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return result_.exit_code();
  }
  [[nodiscard]] constexpr short revents() const noexcept {
    return result_.revents();
  }

private:
  friend class IoAwaiter;
  friend class ::rund::detail::task::OpAccess;

  constexpr IoOp(const IoResult result, const bool deferred, const int fd,
                 const short interest, const std::uint64_t host_handle_id,
                 const std::uint64_t fd_generation) noexcept
      : result_(result), host_handle_id_(host_handle_id),
        fd_generation_(fd_generation), fd_(fd), interest_(interest),
        deferred_(deferred) {}

  IoResult result_{};
  std::uint64_t host_handle_id_ = 0u;
  std::uint64_t fd_generation_ = 0u;
  int fd_ = -1;
  short interest_ = 0;
  bool deferred_ = false;
};

template <typename T> class ReceiveResult final {
public:
  ReceiveResult() = default;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value_.has_value();
  }
  [[nodiscard]] bool ok() const noexcept { return value_.has_value(); }
  [[nodiscard]] ReasonCode code() const noexcept {
    return value_ ? ReasonCode::Ok : status_.code();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return value_ ? std::string_view{} : status_.error();
  }
  [[nodiscard]] int exit_code() const noexcept {
    return value_ ? 0 : status_.exit_code();
  }
  [[nodiscard]] T *operator->() noexcept {
    return value_ ? std::addressof(*value_) : nullptr;
  }
  [[nodiscard]] const T *operator->() const noexcept {
    return value_ ? std::addressof(*value_) : nullptr;
  }
  [[nodiscard]] T &operator*() & noexcept { return *value_; }
  [[nodiscard]] const T &operator*() const & noexcept { return *value_; }
  [[nodiscard]] T &&operator*() && noexcept { return std::move(*value_); }

  template <typename U = T>
  [[nodiscard]] static ReceiveResult
  success(U &&value) noexcept(std::is_nothrow_constructible_v<T, U &&>) {
    return ReceiveResult{std::forward<U>(value)};
  }

  [[nodiscard]] static ReceiveResult fail(const ReasonCode code) noexcept {
    return ReceiveResult{Status::fail(code)};
  }

private:
  template <typename> friend class channel;

  template <typename U>
  explicit ReceiveResult(U &&value) noexcept(
      std::is_nothrow_constructible_v<T, U &&>)
      : value_(std::in_place, std::forward<U>(value)),
        status_(Status::success()) {}

  explicit ReceiveResult(const Status status) noexcept : status_(status) {}

  std::optional<T> value_{};
  Status status_{};
};

[[nodiscard]] YieldOp yield() noexcept;
[[nodiscard]] SleepOp sleep(std::chrono::nanoseconds duration) noexcept;

} // namespace rund::task

namespace rund::detail::task {

inline ::rund::task::YieldOp OpAccess::yield(const ::rund::task::Status status,
                                             const bool deferred) noexcept {
  return ::rund::task::YieldOp{status, deferred};
}

inline ::rund::task::SleepOp
OpAccess::sleep(const ::rund::task::Status status,
                const std::chrono::nanoseconds duration,
                const bool deferred) noexcept {
  return ::rund::task::SleepOp{status, duration, deferred};
}

inline ::rund::task::IoOp
OpAccess::io(const ::rund::task::Status status, const short revents,
             const bool deferred, const int fd, const short interest,
             const std::uint64_t host_handle_id,
             const std::uint64_t fd_generation) noexcept {
  return ::rund::task::IoOp{::rund::task::IoResult{status, revents},
                            deferred,
                            fd,
                            interest,
                            host_handle_id,
                            fd_generation};
}

} // namespace rund::detail::task
