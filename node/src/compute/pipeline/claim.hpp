#pragma once

#include "state.hpp"

#include <cstddef>
#include <span>

namespace rund::compute::detail {

[[nodiscard]] Status acquire_claims(DeviceState &device,
                                    std::span<const BufferClaim> claims,
                                    bool reject_poison = true) noexcept;
void release_claims(DeviceState &device,
                    std::span<const BufferClaim> claims) noexcept;
// Publishes write generation or poison and releases the complete claim set in
// one Device claim-gate acquisition. `poison_writes` applies only on failure.
void publish_claims(DeviceState &device, std::span<const BufferClaim> claims,
                    bool succeeded, bool poison_writes) noexcept;
[[nodiscard]] bool buffer_poisoned(const BufferState &buffer) noexcept;
// One device-affinity predicate for every prepare-time Pipeline resource path.
// Callers may order this check at their existing admission boundary, but may
// not reconstruct owner/device equality locally.
[[nodiscard]] Status
validate_pipeline_resource_device(const PipelineState &state,
                                  const PipelineResource &resource) noexcept;
[[nodiscard]] Status
validate_pipeline_resources(const PipelineState &state) noexcept;
[[nodiscard]] Status acquire_pipeline_claims(PipelineState &state) noexcept;
void close_pipeline_observation_epoch(PipelineState &state) noexcept;
void synchronize_pipeline_observation_epoch(
    PipelineState &state, const PipelinePublicationState &publication) noexcept;

struct PipelineTerminal final {
  Reason reason{Reason::Ok};
  std::size_t verified{};
  std::size_t failed_step{};
  bool failure_step_known{};
  bool writes_possible{};
  // True only when CPU never entered publication or a valid device control
  // proved the success-gated publication shader took its no-store path.
  bool publication_suppressed{};
};

// Publishes the terminal Pipeline state and releases every resource claim.
// Write poison/generation publication happens under the same Device claim-gate
// acquisition and before the corresponding writer claim is released.
void publish_pipeline_terminal(PipelineState &state,
                               PipelineTerminal terminal) noexcept;

class ClaimGuard final {
public:
  ClaimGuard(DeviceState &device,
             const std::span<const BufferClaim> claims) noexcept
      : device_(&device), claims_(claims) {}
  ClaimGuard(const ClaimGuard &) = delete;
  ClaimGuard &operator=(const ClaimGuard &) = delete;
  ~ClaimGuard() {
    if (device_ != nullptr) {
      release_claims(*device_, claims_);
    }
  }
  void release() noexcept {
    if (device_ != nullptr) {
      release_claims(*device_, claims_);
      device_ = nullptr;
    }
  }
  // Ownership was discharged by publish_claims under its publication lock.
  void dismiss() noexcept { device_ = nullptr; }

private:
  DeviceState *device_{};
  std::span<const BufferClaim> claims_{};
};

} // namespace rund::compute::detail
