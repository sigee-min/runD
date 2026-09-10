#include "internal.hpp"

#include <atomic>
#include <limits>

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

rund::AccelCheck Begin(const std::shared_ptr<void> &prepared,
                       const KernelCompletion completion, void *const user,
                       const KernelTiming timing, const PipelineSubmitMode mode,
                       MetalSequence *&sequence) noexcept {
  sequence = static_cast<MetalSequence *>(prepared.get());
  if (!ValidMetalSequence(sequence) || sequence->adapter == nullptr ||
      !sequence->residency_sliding.ready_for_submit ||
      sequence->residency_sliding.quarantined.load(std::memory_order_acquire) ||
      completion == nullptr || user == nullptr ||
      timing != KernelTiming::Submission ||
      mode != PipelineSubmitMode::Residency) {
    return {false, "accel_metal_command_unavailable"};
  }
  if (!submission::Begin(sequence->submission, *sequence, completion, user)) {
    return {false, "compute_pipeline_busy"};
  }
  return {true, "ok"};
}

rund::AccelCheck Validate(const Context &context) noexcept {
  if (context.adapter.residency_quarantined.load(std::memory_order_acquire) ||
      !context.native.ready() || context.native.terminal.active() ||
      context.gate.active || context.gate.ready == nil ||
      context.queue == nil || context.guard == nullptr ||
      context.destination == nullptr || context.gate.next_ready_value == 0u ||
      context.gate.next_ready_value ==
          std::numeric_limits<std::uint64_t>::max() ||
      context.native.event_value == TerminalCell::MaxGeneration) {
    return {false, "accel_metal_command_unavailable"};
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
