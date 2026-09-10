#include "internal.hpp"

#include <atomic>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

std::shared_ptr<void>
FinalizeMetalResidencySliding(MetalSequence &sequence,
                              KernelResult &result) noexcept {
  MetalResidencySlidingGate &gate = sequence.residency_sliding;
  std::shared_ptr<void> retained = std::move(gate.active_owner);
  bool accepted = false;
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  // UnknownMayWrite has no GPU-to-Host acquire edge: the GPU may still own
  // these result words. Read them only after the exact Known shared-event
  // terminal. The caller holds residency_terminal_gate across this entire
  // classification and the sticky quarantine publication below.
  if (result.terminal == NativeTerminal::Known && result.check.ok) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      if (gate.descriptor != nil &&
          gate.descriptor.length >= sizeof(MetalResidencySlidingPayload)) {
        const auto *const payload =
            static_cast<const MetalResidencySlidingPayload *>(
                [gate.descriptor contents]);
        if (payload != nullptr) {
          gate.gpu_result_read_count.fetch_add(1u, std::memory_order_relaxed);
          const std::uint64_t expected_generation =
              static_cast<std::uint64_t>(
                  payload->words
                      [metal_residency_sliding::DescriptorGenerationWord]) |
              (static_cast<std::uint64_t>(
                   payload->words
                       [metal_residency_sliding::DescriptorGenerationWord + 1u])
               << 32u);
          const std::uint64_t observed_generation =
              static_cast<std::uint64_t>(
                  payload->words[MetalResidencySlidingObservedGenerationWord]) |
              (static_cast<std::uint64_t>(
                   payload->words[MetalResidencySlidingObservedGenerationWord +
                                  1u])
               << 32u);
          accepted = payload->words[MetalResidencySlidingAcceptedWord] == 1u &&
                     payload->words[MetalResidencySlidingReasonWord] == 0u &&
                     expected_generation != 0u &&
                     observed_generation == expected_generation;
        }
      }
    }
  }
#endif
  gate.active = false;
  if (!result.check.ok || !accepted) {
    if (result.check.ok) {
      result.check = {false, "accel_kernel_pipeline_invalid"};
    }
    result.terminal = NativeTerminal::UnknownMayWrite;
    result.stats.outcome = {.ok = false, .reason = result.check.reason};
    gate.quarantined.store(true, std::memory_order_release);
    gate.quarantine_owner = std::move(retained);
    sequence.residency_submission.phase =
        MetalResidencySubmissionPhase::Quarantined;
    sequence.adapter->residency_quarantined.store(true,
                                                  std::memory_order_release);
    SetMetalLastError(*sequence.adapter, "compute_device_lost");
  }
  return retained;
}

#endif

} // namespace rund::node::accel::detail
