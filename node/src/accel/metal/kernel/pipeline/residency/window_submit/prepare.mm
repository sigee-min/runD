#include "internal.hpp"

#include "../encode.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

const std::uint64_t MetalResidencyWindowTerminalDeadlineNs = 30'000'000'000u;
const std::uint64_t MetalResidencyWindowFaultDeadlineNs = 10'000'000u;

rund::AccelCheck PrepareMetalResidencyWindowCommand(
    MetalSequence &sequence, MetalResidencyWindowCommand &command,
    const std::span<const std::uint32_t> locals) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  MetalResidencyWindow &window = sequence.residency_window;
  MetalResidencySubmission &residency = sequence.residency_submission;
  if (!window.ready_for_submit() || !residency.ready() ||
      command.allocator == nil || command.command == nil ||
      command.watchdog == nil || command.terminal.active() || locals.empty()) {
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  [command.command beginCommandBufferWithAllocator:command.allocator];
  [command.command useResidencySet:residency.resources];
  id<MTL4ComputeCommandEncoder> const encoder =
      [command.command computeCommandEncoder];
  const rund::AccelCheck encoded = EncodeMetalResidencySelection(
      encoder, sequence, locals, command.dispatch_count, command.control_count,
      command.reset_count, command.reset_bytes);
  if (encoder != nil) {
    [encoder endEncoding];
  }
  [command.command endCommandBuffer];
  if (!encoded.ok) {
    [command.allocator reset];
  }
  return encoded;
}

void CancelMetalResidencyWindowCommand(
    MetalResidencyWindowCommand &command) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  [command.allocator reset];
  command.leader.reset();
  command.terminal_generation = 0u;
  command.ready_value = 0u;
  command.done_value = 0u;
  command.dispatch_count = 0u;
  command.control_count = 0u;
  command.reset_count = 0u;
  command.reset_bytes = 0u;
  command.batch_index = 0u;
  command.admission = rund::AccelCheck{true, "ok"};
  command.force_device_lost = false;
  command.force_terminal_loss = false;
  command.force_abort_unknown = false;
}

#endif

#endif

} // namespace rund::node::accel::detail
