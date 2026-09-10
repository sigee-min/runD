#include "internal.hpp"

#include <atomic>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

bool InjectMetalResidencySlidingStaleDescriptorOnce(
    const std::shared_ptr<void> &prepared) noexcept {
  auto *const sequence = static_cast<MetalSequence *>(prepared.get());
  if (!ValidMetalSequence(sequence) ||
      !sequence->residency_sliding.ready_for_submit ||
      sequence->residency_sliding.quarantined.load(std::memory_order_acquire)) {
    return false;
  }
  sequence->residency_sliding.force_stale_once.store(true,
                                                     std::memory_order_release);
  return true;
}

bool InspectMetalResidencySliding(
    const std::shared_ptr<void> &prepared,
    MetalResidencySlidingDiagnostics &diagnostics) noexcept {
  diagnostics = {};
  auto *const sequence = static_cast<MetalSequence *>(prepared.get());
  if (!ValidMetalSequence(sequence) || sequence->adapter == nullptr) {
    return false;
  }
  MetalResidencySlidingGate &gate = sequence->residency_sliding;
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    diagnostics.descriptor_identity =
        reinterpret_cast<std::uintptr_t>((__bridge void *)gate.descriptor);
    diagnostics.pipeline_identity =
        reinterpret_cast<std::uintptr_t>((__bridge void *)gate.pipeline);
    diagnostics.gate_command_identity =
        reinterpret_cast<std::uintptr_t>((__bridge void *)gate.command);
    diagnostics.ready_event_identity =
        reinterpret_cast<std::uintptr_t>((__bridge void *)gate.ready);
    diagnostics.command_identity = reinterpret_cast<std::uintptr_t>(
        (__bridge void *)sequence->residency_submission.command);
    diagnostics.allocator_identity = reinterpret_cast<std::uintptr_t>(
        (__bridge void *)sequence->residency_submission.allocator);
  }
#endif
  {
    std::lock_guard lock{sequence->adapter->mutex};
    diagnostics.pipeline_compile_count =
        sequence->adapter->stats.runtime.run.allocations.pipeline_compile_count;
    diagnostics.buffer_allocation_count =
        sequence->adapter->stats.runtime.run.allocations
            .buffer_allocation_count;
    diagnostics.command_submit_count =
        sequence->adapter->stats.runtime.run.work.command_submit_count;
  }
  diagnostics.command_active = sequence->adapter->residency_command_active.load(
      std::memory_order_acquire);
  diagnostics.retained_bytes = gate.retained_bytes;
  diagnostics.gpu_result_read_count =
      gate.gpu_result_read_count.load(std::memory_order_relaxed);
  diagnostics.ready = gate.ready_for_submit;
  diagnostics.quarantined = gate.quarantined.load(std::memory_order_acquire);
  return true;
}

#endif

} // namespace rund::node::accel::detail
