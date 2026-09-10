#include "encode.hpp"
#include "local.hpp"

#include "../../../../clock.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

constexpr std::uint64_t TerminalDeadlineNs = 30u * NSEC_PER_SEC;
constexpr std::uint64_t FaultTerminalDeadlineNs = 10u * NSEC_PER_MSEC;

[[nodiscard]] rund::AccelCheck
PrepareMetalResidencyCommand(MetalSequence &sequence,
                             const std::span<const std::uint32_t> locals,
                             const std::uint64_t submit_begin) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  const rund::AccelCheck unavailable{false, "accel_metal_command_unavailable"};
  MetalResidencySubmission &submission = sequence.residency_submission;
  if (!submission.ready() || MetalResidencyQueue(sequence) == nil ||
      submission.event == nil || submission.listener == nil ||
      submission.watchdog == nil || submission.terminal.active()) {
    return unavailable;
  }
  [submission.command beginCommandBufferWithAllocator:submission.allocator];
  [submission.command useResidencySet:submission.resources];
  id<MTL4ComputeCommandEncoder> const encoder =
      [submission.command computeCommandEncoder];
  std::uint64_t dispatch_count = 0u;
  std::uint64_t control_count = 0u;
  std::uint64_t reset_count = 0u;
  std::uint64_t reset_bytes = 0u;
  const rund::AccelCheck encoded =
      EncodeMetalResidencySelection(encoder, sequence, locals, dispatch_count,
                                    control_count, reset_count, reset_bytes);
  if (encoder != nil) {
    [encoder endEncoding];
  }
  [submission.command endCommandBuffer];
  if (!encoded.ok) {
    [submission.allocator reset];
    submission.phase = MetalResidencySubmissionPhase::Failed;
    return encoded;
  }
  submission.submit_begin_ns = submit_begin;
  submission.dispatch_count = dispatch_count;
  submission.control_count = control_count;
  submission.reset_count = reset_count;
  submission.reset_bytes = reset_bytes;
  submission.native_submit_count = 1u;
  submission.command_inflight_peak = 0u;
  submission.force_device_lost = false;
  submission.force_terminal_loss = false;
  return rund::AccelCheck{true, "ok"};
}

void CancelPreparedMetalResidencyCommand(MetalSequence &sequence) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  MetalResidencySubmission &submission = sequence.residency_submission;
  [submission.allocator reset];
}

[[nodiscard]] bool
NextMetalResidencyGeneration(MetalSequence &sequence,
                             std::uint64_t &generation) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  MetalResidencySubmission &submission = sequence.residency_submission;
  if (submission.event_value == TerminalCell::MaxGeneration ||
      submission.terminal.active()) {
    return false;
  }
  // Hide a preceding dispatch-source event from the newly armed generation
  // until this submit publishes its own exact deadline.
  submission.watchdog_deadline_ns.store(0u, std::memory_order_release);
  generation = ++submission.event_value;
  return submission.terminal.arm(generation);
}

void ObserveMetalResidencyTerminal(MetalSequence &sequence,
                                   const std::uint64_t generation) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  MetalResidencySubmission &submission = sequence.residency_submission;
  const std::weak_ptr<void> owner = submission.owner;
  [submission.event
      notifyListener:submission.listener
             atValue:generation
               block:^(id<MTLSharedEvent>, std::uint64_t value) {
                 const std::shared_ptr<void> retained = owner.lock();
                 if (retained == nullptr || value < generation) {
                   return;
                 }
                 CompleteMetalResidencySubmission(
                     *static_cast<MetalSequence *>(retained.get()), generation,
                     NativeTerminal::Known);
               }];
}

#endif

} // namespace

rund::AccelCheck SubmitMetalResidencySubmission(
    MetalSequence &sequence, const KernelTiming timing,
    const std::span<const std::uint32_t> locals) noexcept {
  const rund::AccelCheck unavailable{false, "accel_metal_command_unavailable"};
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (sequence.adapter == nullptr) {
      return rund::AccelCheck{false, "compute_device_lost"};
    }
    MetalAdapter &adapter = *sequence.adapter;
    std::unique_lock terminal_gate{adapter.residency_terminal_gate};
    if (adapter.residency_quarantined.load(std::memory_order_acquire)) {
      return rund::AccelCheck{false, "compute_device_lost"};
    }
    const std::uint64_t submit_begin = MonotonicNanoseconds();
    const rund::AccelCheck prepared =
        PrepareMetalResidencyCommand(sequence, locals, submit_begin);
    if (!prepared.ok) {
      return prepared;
    }
    MetalResidencySubmission &submission = sequence.residency_submission;
    submission.force_device_lost =
        adapter.device_loss_fault.take(SubmitKind::Work);
    submission.force_terminal_loss =
        adapter.fault_residency_terminal_once.exchange(
            false, std::memory_order_acq_rel);
    std::uint64_t generation = 0u;
    if (!NextMetalResidencyGeneration(sequence, generation)) {
      CancelPreparedMetalResidencyCommand(sequence);
      return unavailable;
    }
    if (!submission.force_terminal_loss) {
      ObserveMetalResidencyTerminal(sequence, generation);
    }
    const id<MTL4CommandBuffer> commands[] = {submission.command};
    id<MTL4CommandQueue> const queue = MetalResidencyQueue(sequence);
    [queue commit:commands count:1u];
    submission.command_inflight_peak = BeginMetalResidencyCommand(adapter);
    const std::uint64_t deadline_delta = submission.force_terminal_loss
                                             ? FaultTerminalDeadlineNs
                                             : TerminalDeadlineNs;
    const std::uint64_t deadline =
        submit_begin >
                std::numeric_limits<std::uint64_t>::max() - deadline_delta
            ? std::numeric_limits<std::uint64_t>::max()
            : submit_begin + deadline_delta;
    submission.watchdog_deadline_ns.store(deadline, std::memory_order_release);
    dispatch_source_set_timer(
        submission.watchdog,
        dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(deadline_delta)),
        DISPATCH_TIME_FOREVER, 0u);
    // Arm the exact-generation deadline before publishing the native event.
    // The event callback may run inline; arming afterward could let this
    // generation overwrite a following generation's timer.
    terminal_gate.unlock();
    if (!submission.force_terminal_loss) {
      [queue signalEvent:submission.event value:generation];
    }
    static_cast<void>(timing);
    return rund::AccelCheck{true, "ok"};
  }
#else
  static_cast<void>(sequence);
  static_cast<void>(timing);
#endif
  return unavailable;
}

#endif

} // namespace rund::node::accel::detail
