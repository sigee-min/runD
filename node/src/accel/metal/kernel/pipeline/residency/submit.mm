#include "local.hpp"

#include "../../../../clock.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

[[nodiscard]] rund::AccelCheck
EncodeMetalResidencyIcb(id<MTL4ComputeCommandEncoder> const encoder,
                        const MetalWarmSubmission warm) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  if (encoder == nil || warm.chunks == nullptr || warm.chunk_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  for (NSUInteger index = 0u; index < warm.chunk_count; ++index) {
    const MetalIcbChunk &chunk = warm.chunks[index];
    if (!chunk.valid() || (index == 0u && chunk.barrier_before())) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    if (chunk.barrier_before()) {
      [encoder barrierAfterEncoderStages:MTLStageDispatch
                     beforeEncoderStages:MTLStageDispatch
                       visibilityOptions:MTL4VisibilityOptionDevice];
    }
    [encoder executeCommandsInBuffer:chunk.commands
                           withRange:NSMakeRange(0u, chunk.command_count)];
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace

KernelResult RunMetalResidencySubmission(MetalSequence &sequence,
                                         const KernelTiming timing) noexcept {
  KernelResult result{
      .check = {false, "accel_metal_command_unavailable"},
      .stats =
          rund::RuntimeStats{
              .outcome = {.ok = false,
                          .reason = "accel_metal_command_unavailable"}},
  };
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    MetalResidencySubmission &submission = sequence.residency_submission;
    if (!submission.ready() || sequence.adapter == nullptr ||
        submission.event_value == std::numeric_limits<std::uint64_t>::max()) {
      return result;
    }
    ++submission.event_value;
    const std::uint64_t submit_begin = MonotonicNanoseconds();
    const bool force_device_lost =
        sequence.adapter->fault_device_lost_once.exchange(
            false, std::memory_order_relaxed);
    [submission.command beginCommandBufferWithAllocator:submission.allocator];
    [submission.command useResidencySet:submission.resources];
    id<MTL4ComputeCommandEncoder> const encoder =
        [submission.command computeCommandEncoder];
    const rund::AccelCheck encoded =
        EncodeMetalResidencyIcb(encoder, sequence.warm);
    if (encoder != nil) {
      [encoder endEncoding];
    }
    [submission.command endCommandBuffer];
    if (!encoded.ok) {
      [submission.allocator reset];
      submission.phase = MetalResidencySubmissionPhase::Failed;
      result.check = encoded;
      result.stats.outcome.reason = encoded.reason;
      return result;
    }
    const id<MTL4CommandBuffer> commands[] = {submission.command};
    [submission.queue commit:commands count:1u];
    [submission.queue signalEvent:submission.event
                            value:submission.event_value];
    const bool event_complete =
        [submission.event waitUntilSignaledValue:submission.event_value
                                       timeoutMS:5000u];
    if (event_complete) {
      [submission.allocator reset];
    }
    const std::uint64_t submit_wait_ns = MonotonicNanoseconds() - submit_begin;
    result.stats.run.work.command_submit_count = 1u;
    result.stats.run.time.command_submit_wait_ns = submit_wait_ns;
    if (force_device_lost) {
      result.check = rund::AccelCheck{false, "compute_device_lost"};
    } else if (!event_complete) {
      result.check = rund::AccelCheck{false, "accel_metal_command_unavailable"};
    } else {
      result.check = rund::AccelCheck{true, "ok"};
      static_cast<void>(timing);
    }
    result.stats.outcome.ok = result.check.ok;
    result.stats.outcome.reason = result.check.reason;
    if (!result.check.ok) {
      submission.phase = MetalResidencySubmissionPhase::Failed;
    }
  }
#else
  static_cast<void>(sequence);
  static_cast<void>(timing);
#endif
  return result;
}

#endif

} // namespace rund::node::accel::detail
