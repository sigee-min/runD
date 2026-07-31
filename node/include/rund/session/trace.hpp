#pragma once

#include <rund/compute/reason.hpp>
#include <rund/reason.hpp>
#include <rund/session/state.hpp>

#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rund {

enum class TraceEvent : std::uint8_t {
  RuntimeConfigured,
  RuntimeStarted,
  RuntimeDraining,
  RuntimeStopped,
  ComputeSubmitted,
  ComputeAdmitted,
  ComputeDispatchStarted,
  ComputeBackendSubmitted,
  ComputeCompleted,
  TelemetryEmitted,
  TelemetrySkipped,
};

enum class TraceDomain : std::uint16_t {
  Runtime,
  Compute,
};

class TraceCode final {
public:
  constexpr TraceCode() noexcept = default;

  [[nodiscard]] static constexpr TraceCode
  runtime(const ReasonCode code) noexcept {
    return TraceCode{TraceDomain::Runtime, static_cast<std::uint16_t>(code)};
  }

  [[nodiscard]] static constexpr TraceCode
  compute(const ::rund::compute::Reason reason) noexcept {
    return TraceCode{TraceDomain::Compute, static_cast<std::uint16_t>(reason)};
  }

  [[nodiscard]] constexpr TraceDomain domain() const noexcept {
    return domain_;
  }

  [[nodiscard]] constexpr std::uint16_t value() const noexcept {
    return value_;
  }

  [[nodiscard]] constexpr bool ok() const noexcept { return value_ == 0u; }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::string_view error() const noexcept;
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ok() ? 0 : 1;
  }

  [[nodiscard]] constexpr std::optional<ReasonCode>
  runtime_code() const noexcept {
    return domain_ == TraceDomain::Runtime
               ? std::optional<ReasonCode>{static_cast<ReasonCode>(value_)}
               : std::nullopt;
  }

  [[nodiscard]] constexpr std::optional<::rund::compute::Reason>
  compute_reason() const noexcept {
    return domain_ == TraceDomain::Compute
               ? std::optional<::rund::compute::Reason>{static_cast<
                     ::rund::compute::Reason>(value_)}
               : std::nullopt;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const TraceCode &, const TraceCode &) noexcept = default;

private:
  constexpr TraceCode(const TraceDomain domain,
                      const std::uint16_t value) noexcept
      : domain_(domain), value_(value) {}

  TraceDomain domain_ = TraceDomain::Runtime;
  std::uint16_t value_ = 0u;
};

static_assert(sizeof(TraceCode) == 4u);
static_assert(std::is_trivially_copyable_v<TraceCode>);

struct TraceRecord final {
  struct State final {
    SessionState state = SessionState::Unconfigured;
    std::uint32_t active_compute_jobs = 0u;
    bool scope_active = false;
    ReasonCode code = ReasonCode::Ok;

    [[nodiscard]] constexpr bool ok() const noexcept {
      return code == ReasonCode::Ok;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
      return ok();
    }
    [[nodiscard]] std::string_view error() const noexcept {
      return ok() ? std::string_view{} : ReasonString(code);
    }
    [[nodiscard]] constexpr int exit_code() const noexcept {
      return ok() ? 0 : 1;
    }
  };

  ::rund::TraceEvent event = ::rund::TraceEvent::RuntimeConfigured;
  TraceCode code{};
  State snapshot{};
  std::uint64_t sequence = 0u;

  [[nodiscard]] constexpr bool ok() const noexcept { return code.ok(); }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept { return code.error(); }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return code.exit_code();
  }
};

struct Trace final {
  ReasonCode code = ReasonCode::Ok;
  std::uint64_t dropped = 0u;
  std::vector<::rund::TraceRecord> records{};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ReasonCode::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ok() ? std::string_view{} : ReasonString(code);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ok() ? 0 : 1;
  }
};

} // namespace rund
