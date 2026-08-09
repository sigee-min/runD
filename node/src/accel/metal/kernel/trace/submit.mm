#include "submit.hpp"

#include "../../command/submit.hpp"

#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

void FinishTraceSubmission(MetalDispatchTrace &trace,
                           KernelResult result) noexcept {
  KernelCompletion const completion = trace.submit_completion;
  void *const user = trace.submit_user;
  result.stats.run.work.command_submit_count = trace.submitted_commands;
  trace.submit_adapter = nullptr;
  trace.submit_completion = nullptr;
  trace.submit_user = nullptr;
  trace.submitted_commands = 0u;
  if (completion != nullptr) {
    completion(user, result);
  }
}

void CompleteTraceResolve(void *const raw, KernelResult result) noexcept {
  auto *const trace = static_cast<MetalDispatchTrace *>(raw);
  if (trace == nullptr) {
    return;
  }
  ::rund::detail::counter::Accumulate(
      trace->submitted_commands, result.stats.run.work.command_submit_count);
  FinishTraceSubmission(*trace, result);
}

void CompleteTraceSamples(void *const raw, KernelResult result) noexcept {
  auto *const trace = static_cast<MetalDispatchTrace *>(raw);
  if (trace == nullptr) {
    return;
  }
  ::rund::detail::counter::Accumulate(
      trace->submitted_commands, result.stats.run.work.command_submit_count);
  if (!result.check.ok || trace->submit_adapter == nullptr) {
    FinishTraceSubmission(*trace, result);
    return;
  }
  id<MTLCommandQueue> const queue =
      (__bridge id<MTLCommandQueue>)trace->submit_adapter->queue.get();
  id<MTLCommandBuffer> const command =
      queue == nil ? nil : [queue commandBufferWithUnretainedReferences];
  id<MTLBlitCommandEncoder> const blit =
      command == nil ? nil : [command blitCommandEncoder];
  if (blit == nil) {
    FinishTraceSubmission(
        *trace, KernelResult{.check = rund::AccelCheck{
                                 false, "accel_metal_command_unavailable"}});
    return;
  }
  [blit resolveCounters:trace->samples
                inRange:NSMakeRange(0u, trace->sample_count)
      destinationBuffer:trace->values
      destinationOffset:0u];
  [blit endEncoding];
  if (trace->submit_adapter->fault_trace_resolve_device_lost_once.exchange(
          false, std::memory_order_relaxed)) {
    trace->submit_adapter->fault_device_lost_once.store(
        true, std::memory_order_relaxed);
  }
  const rund::AccelCheck queued =
      QueueCommand(*trace->submit_adapter, (__bridge void *)command,
                   CompleteTraceResolve, trace, false);
  if (!queued.ok) {
    FinishTraceSubmission(*trace, KernelResult{.check = queued});
  }
}

} // namespace

rund::AccelCheck QueueMetalDispatchTrace(
    MetalAdapter &adapter, id<MTLCommandBuffer> const sampled_command,
    MetalDispatchTrace &trace, const KernelCompletion completion,
    void *const user) noexcept {
  if (sampled_command == nil || !trace.available() || trace.failed ||
      completion == nullptr || trace.submit_completion != nullptr) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  trace.submit_adapter = &adapter;
  trace.submit_completion = completion;
  trace.submit_user = user;
  trace.submitted_commands = 0u;
  const rund::AccelCheck queued =
      QueueCommand(adapter, (__bridge void *)sampled_command,
                   CompleteTraceSamples, &trace, false);
  if (!queued.ok) {
    trace.submit_adapter = nullptr;
    trace.submit_completion = nullptr;
    trace.submit_user = nullptr;
  }
  return queued;
}

#endif

} // namespace rund::node::accel::detail
