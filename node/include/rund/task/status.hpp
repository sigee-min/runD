#pragma once

#include <rund/reason.hpp>

#include <string_view>

namespace rund::task {

enum class Phase : std::uint8_t {
  Idle,
  Admitted,
  Ready,
  Running,
  Parked,
  Committing,
  Completed,
  Failed,
  Cancelled,
};

struct Poll final {
  Phase phase{Phase::Idle};
  ReasonCode code{ReasonCode::Ok};

  [[nodiscard]] constexpr bool terminal() const noexcept {
    return phase == Phase::Completed || phase == Phase::Failed ||
           phase == Phase::Cancelled;
  }
};

class Status final {
public:
  constexpr Status() noexcept = default;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return code_ == ReasonCode::Ok;
  }

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code_ == ReasonCode::Ok;
  }

  [[nodiscard]] constexpr ReasonCode code() const noexcept { return code_; }

  [[nodiscard]] std::string_view error() const noexcept {
    return ok() ? std::string_view{} : std::string_view{ReasonString(code_)};
  }

  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ok() ? 0 : 1;
  }

  [[nodiscard]] static constexpr Status success() noexcept {
    return Status{ReasonCode::Ok};
  }

  [[nodiscard]] static constexpr Status
  fail(const ReasonCode code) noexcept {
    return Status{code == ReasonCode::Ok ? ReasonCode::TaskFailed : code};
  }

private:
  explicit constexpr Status(const ReasonCode code) noexcept : code_(code) {}

  ReasonCode code_{ReasonCode::TaskInvalid};
};

} // namespace rund::task
