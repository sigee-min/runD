#pragma once

#include <rund/task/cancel/identity.hpp>
#include <rund/task/status.hpp>

#include <cstdint>
#include <string_view>

namespace rund::detail::task {
class StopAccess;
} // namespace rund::detail::task

namespace rund::task {

struct Handle;

class StopState final {
public:
  constexpr StopState() noexcept = default;

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
  [[nodiscard]] constexpr bool requested() const noexcept { return requested_; }

  [[nodiscard]] static constexpr StopState
  success(const bool requested) noexcept {
    return StopState{Status::success(), requested};
  }
  [[nodiscard]] static constexpr StopState
  fail(const ReasonCode code) noexcept {
    return StopState{Status::fail(code), false};
  }

private:
  constexpr StopState(const Status status, const bool requested) noexcept
      : status_(status), requested_(requested) {}

  Status status_{};
  bool requested_ = false;
};

class stop_token {
public:
  constexpr stop_token() noexcept = default;
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return identity_.valid();
  }
  [[nodiscard]] StopState state() const noexcept;

private:
  friend class stop_source;
  friend class ::rund::detail::task::StopAccess;
  explicit constexpr stop_token(
      const ::rund::detail::task::StopIdentity identity) noexcept
      : identity_(identity) {}
  ::rund::detail::task::StopIdentity identity_{};
};

static_assert(sizeof(stop_token) == sizeof(::rund::detail::task::StopIdentity));

class stop_source {
public:
  stop_source() noexcept = default;
  [[nodiscard]] static stop_source create() noexcept;
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return static_cast<bool>(token_);
  }
  [[nodiscard]] stop_token token() const noexcept { return token_; }
  [[nodiscard]] Status request_stop() const noexcept;

private:
  explicit constexpr stop_source(stop_token token) noexcept : token_(token) {}
  stop_token token_{};
};

} // namespace rund::task
