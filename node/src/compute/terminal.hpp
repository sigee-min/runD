#pragma once

#include "memory/profile.hpp"

#include <utility>
#include <variant>

namespace rund::compute::detail {

// One allocation-free terminal handoff from a Job or Pipeline owner to the
// Session coordinator. Construction is restricted to the owner-local finish
// path so status, execution, and memory cannot name different run epochs.
class TerminalObservation final {
public:
  [[nodiscard]] Status status() const noexcept { return status_; }
  [[nodiscard]] const Stats &stats() const noexcept {
    const auto *const profile = std::get_if<telemetry::Profile>(&evidence_);
    return profile == nullptr ? std::get<Stats>(evidence_)
                              : profile->execution();
  }
  [[nodiscard]] const telemetry::Profile *profile() const noexcept {
    return std::get_if<telemetry::Profile>(&evidence_);
  }

private:
  friend struct TerminalObservationAccess;

  TerminalObservation(const Status status, telemetry::Profile profile) noexcept
      : status_(status), evidence_(std::move(profile)) {}

  TerminalObservation(const Status status, const Stats stats) noexcept
      : status_(status), evidence_(stats) {}

  Status status_{Status::fail(Reason::CompletionInvalid)};
  std::variant<Stats, telemetry::Profile> evidence_{Stats{}};
};

struct TerminalObservationAccess final {
  [[nodiscard]] static TerminalObservation
  from_profile(const Status status, std::shared_ptr<const DeviceInfo> device,
               const Stats stats, const MemoryStats memory) noexcept {
    if (device == nullptr) {
      return from_stats(status, stats);
    }
    return TerminalObservation{
        status, ProfileAccess::make(std::move(device), stats, memory)};
  }

  [[nodiscard]] static TerminalObservation
  from_stats(const Status status, const Stats stats = {}) noexcept {
    return TerminalObservation{status, stats};
  }
};

} // namespace rund::compute::detail
