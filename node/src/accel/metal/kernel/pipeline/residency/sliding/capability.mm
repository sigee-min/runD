#include "internal.hpp"

#include <atomic>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

BackendResidencySlidingCapability
MetalResidencySlidingCapability(const std::shared_ptr<void> &prepared,
                                const ResidencySlidingMemory memory) noexcept {
  bool ready = false;
  const rund::AccelCheck checked = MetalPipelineResidencyReady(prepared, ready);
  auto *const sequence = static_cast<MetalSequence *>(prepared.get());
  const bool physical =
      checked.ok && ready && ValidMetalSequence(sequence) &&
      sequence->residency_sliding.ready_for_submit &&
      !sequence->residency_sliding.quarantined.load(
          std::memory_order_acquire) &&
      !sequence->adapter->residency_quarantined.load(std::memory_order_acquire);
  if (!physical || memory != ResidencySlidingMemory::HostCoherent) {
    return BackendResidencySlidingCapability{
        .check = {false, checked.ok ? "accel_metal_command_unavailable"
                                    : checked.reason}};
  }
  return BackendResidencySlidingCapability{
      .check = {true, "ok"},
      .memory = ResidencySlidingMemory::HostCoherent,
      .retained_bytes = sequence->residency_sliding.retained_bytes,
      .transient_bytes = sizeof(MetalResidencySlidingSubmissionStorage),
      .max_slots = static_cast<std::uint8_t>(ResidencySlidingCapacity),
      .callbacks_async = true,
      // Physical source-private execution is proven by the backend contract,
      // but common product admission remains closed until Authority W2..4
      // Release/Final and publication exercise this exact gate end to end.
      .descriptor_release_acquire = false,
      .flush_before_frontier = false,
      .integrated_copy = false,
      .distinct_transfer_queue = false,
  };
}

#endif

} // namespace rund::node::accel::detail
