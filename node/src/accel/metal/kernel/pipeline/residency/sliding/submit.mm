#include "submit/internal.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck SubmitMetalResidencySliding(
    const std::shared_ptr<void> &prepared,
    const BackendResidencySlidingDescriptor &descriptor,
    const KernelCompletion completion, void *const user,
    const KernelTiming timing, const PipelineSubmitMode mode,
    const std::span<const std::uint32_t> locals) noexcept {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    MetalSequence *sequence = nullptr;
    const rund::AccelCheck begun = metal_residency_sliding::submit::Begin(
        prepared, completion, user, timing, mode, sequence);
    if (!begun.ok) {
      return begun;
    }

    std::unique_lock terminal_gate{sequence->adapter->residency_terminal_gate};
    auto context = metal_residency_sliding::submit::Capture(
        prepared, descriptor, locals, *sequence);

    rund::AccelCheck checked =
        metal_residency_sliding::submit::Validate(context);
    if (!checked.ok) {
      return metal_residency_sliding::submit::Cancel(context, terminal_gate,
                                                     checked);
    }
    checked = metal_residency_sliding::submit::Build(context);
    if (!checked.ok) {
      return metal_residency_sliding::submit::Cancel(context, terminal_gate,
                                                     checked);
    }
    checked = metal_residency_sliding::submit::Encode(context);
    if (!checked.ok) {
      return metal_residency_sliding::submit::Cancel(context, terminal_gate,
                                                     checked);
    }
    checked = metal_residency_sliding::submit::ArmTerminal(context);
    if (!checked.ok) {
      return metal_residency_sliding::submit::Cancel(context, terminal_gate,
                                                     checked);
    }

    metal_residency_sliding::submit::Activate(context);
    metal_residency_sliding::submit::InstallTerminalListener(context);
    metal_residency_sliding::submit::PublishDescriptor(context);
    metal_residency_sliding::submit::CommitQueue(context);
    metal_residency_sliding::submit::ArmWatchdog(context);
    terminal_gate.unlock();
    metal_residency_sliding::submit::SignalKnownTerminal(context);
    return {true, "ok"};
  }
#else
  static_cast<void>(prepared);
  static_cast<void>(descriptor);
  static_cast<void>(completion);
  static_cast<void>(user);
  static_cast<void>(timing);
  static_cast<void>(mode);
  static_cast<void>(locals);
#endif
  return {false, "accel_metal_command_unavailable"};
}

#endif

} // namespace rund::node::accel::detail
