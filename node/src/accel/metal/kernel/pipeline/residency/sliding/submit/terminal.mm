#include "internal.hpp"

#include <atomic>

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

rund::AccelCheck ArmTerminal(Context &context) noexcept {
  context.native.watchdog_deadline_ns.store(0u, std::memory_order_release);
  context.terminal_generation = ++context.native.event_value;
  if (!context.native.terminal.arm(context.terminal_generation)) {
    [context.native.allocator reset];
    return {false, "accel_metal_command_unavailable"};
  }
  return {true, "ok"};
}

void Activate(Context &context) noexcept {
  context.native.force_device_lost =
      context.adapter.device_loss_fault.take(SubmitKind::Work);
  context.native.force_terminal_loss =
      context.adapter.fault_residency_terminal_once.exchange(
          false, std::memory_order_acq_rel);
  context.native.submit_begin_ns = context.submit_begin;
  context.native.native_submit_count = 1u;
  context.native.command_inflight_peak = 0u;
  context.gate.active_owner = context.prepared;
  context.gate.active = true;
  metal_residency_sliding::CommitDescriptor(context.gate, context.descriptor);
}

void InstallTerminalListener(Context &context) noexcept {
  if (context.native.force_terminal_loss) {
    return;
  }
  const std::uint64_t terminal_generation = context.terminal_generation;
  const std::weak_ptr<void> owner = context.native.owner;
  [context.native.event
      notifyListener:context.native.listener
             atValue:terminal_generation
               block:^(id<MTLSharedEvent>, std::uint64_t value) {
                 const std::shared_ptr<void> retained = owner.lock();
                 if (retained == nullptr || value < terminal_generation) {
                   return;
                 }
                 CompleteMetalResidencySubmission(
                     *static_cast<MetalSequence *>(retained.get()),
                     terminal_generation, NativeTerminal::Known);
               }];
}

void SignalKnownTerminal(Context &context) noexcept {
  if (!context.native.force_terminal_loss) {
    [context.queue signalEvent:context.native.event
                         value:context.terminal_generation];
  }
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
