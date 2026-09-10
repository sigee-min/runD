#pragma once

#include "../execution.hpp"

#include <accel/check.hpp>

#include <cstdint>
#include <span>
#include <utility>

namespace rund::compute::detail {

struct VirtualRunResources;

enum class VirtualDeviceVsmPhase : std::uint8_t {
  Ordinary,
  DeviceVsm,
  Terminal,
};

struct VirtualDeviceVsmCandidate final {
  VirtualDeviceVsmPhase phase{VirtualDeviceVsmPhase::Terminal};
  Status status{Status::fail(Reason::BackendUnsupported)};
  VirtualDeviceVsmRouteProof proof{};
  // Immutable snapshot of the one pre-side-effect admission decision.  Route
  // policy uses only `phase`; this field is diagnostic evidence for a failed
  // required admission and is never re-evaluated downstream.
  const ::rund::AccelCheck admission{false, "compute_backend_unsupported"};

  [[nodiscard]] bool ordinary() const noexcept {
    return phase == VirtualDeviceVsmPhase::Ordinary;
  }
  [[nodiscard]] bool device_vsm() const noexcept {
    return phase == VirtualDeviceVsmPhase::DeviceVsm;
  }
  [[nodiscard]] bool terminal() const noexcept {
    return phase == VirtualDeviceVsmPhase::Terminal;
  }
};

enum class VirtualDeviceVsmPostPhase : std::uint8_t {
  TerminalFailure,
  Ready,
};

enum class VirtualDeviceVsmScope : std::uint8_t {
  Pooled,
  Poolless,
};

struct VirtualDeviceVsmPostStage final {
  VirtualDeviceVsmPostStage() noexcept = default;
  VirtualDeviceVsmPostStage(const VirtualDeviceVsmPostStage &) = delete;
  VirtualDeviceVsmPostStage &
  operator=(const VirtualDeviceVsmPostStage &) = delete;
  VirtualDeviceVsmPostStage(VirtualDeviceVsmPostStage &&other) noexcept
      : prepared(std::move(other.prepared)), status(other.status),
        phase(other.phase), proof(std::move(other.proof)),
        consumed(other.consumed) {
    other.prepared = {};
    other.phase = VirtualDeviceVsmPostPhase::TerminalFailure;
    other.proof = {};
    other.consumed = true;
  }
  VirtualDeviceVsmPostStage &
  operator=(VirtualDeviceVsmPostStage &&other) noexcept {
    if (this != &other) {
      prepared = std::move(other.prepared);
      status = other.status;
      phase = other.phase;
      proof = std::move(other.proof);
      consumed = other.consumed;
      other.prepared = {};
      other.phase = VirtualDeviceVsmPostPhase::TerminalFailure;
      other.proof = {};
      other.consumed = true;
    }
    return *this;
  }

  VirtualExecutionDeviceVsmPrepared prepared{};
  Status status{Status::fail(Reason::BackendUnsupported)};
  VirtualDeviceVsmPostPhase phase{VirtualDeviceVsmPostPhase::TerminalFailure};
  VirtualDeviceVsmRouteProof proof{};
  bool consumed{};

  [[nodiscard]] bool ready() const noexcept {
    return phase == VirtualDeviceVsmPostPhase::Ready && prepared &&
           !consumed;
  }

  [[nodiscard]] bool consume() noexcept {
    if (!ready()) {
      return false;
    }
    consumed = true;
    return true;
  }
};

// Purely selects the route before pooled staging or native preparation. Only
// Ordinary permits the existing fallback; Terminal carries a required-route
// failure without executing or creating an owner.
[[nodiscard]] VirtualDeviceVsmCandidate probe_virtual_device_vsm_route(
    const VirtualPipelineState &, std::span<VirtualBacking *const>,
    const VirtualBacking &, const VirtualRunProjection &) noexcept;

[[nodiscard]] VirtualDeviceVsmCandidate probe_virtual_device_vsm_route(
    const VirtualPipelineState &, VirtualBacking &, const VirtualBacking &,
    const VirtualRunProjection &) noexcept;

// Performs the side-effecting owner preparation for a DeviceVsm candidate.
// BackendUnsupported returned here is terminal; it is never reinterpreted as
// an Ordinary fallback after pooled staging has begun.
[[nodiscard]] VirtualDeviceVsmPostStage prepare_virtual_device_vsm_route(
    VirtualPipelineState &, const VirtualRunProjection &,
    VirtualDeviceVsmCandidate) noexcept;

[[nodiscard]] Status rearm_virtual_device_vsm_route(
    VirtualDeviceVsmPostStage &) noexcept;

[[nodiscard]] Status dispose_virtual_device_vsm_route(
    VirtualPipelineState &, VirtualRunResources &, VirtualDeviceVsmPostStage &,
    VirtualDeviceVsmScope, Status, bool &, VirtualRunWriteCertainty &) noexcept;

// Executes only an unconsumed Ready post-stage. The backend callback remains
// the sole native submit/Final owner and its exact certainty is returned
// unchanged.
[[nodiscard]] VirtualExecutionResult execute_virtual_device_vsm_route(
    VirtualPipelineState &, std::span<VirtualBacking *const>, VirtualBacking &,
    const VirtualRunProjection &, VirtualDeviceVsmPostStage &,
    Stats &) noexcept;

} // namespace rund::compute::detail
