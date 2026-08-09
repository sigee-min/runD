#include "submit.hpp"

#include "../../command/submit.hpp"

#include <rund/counter.hpp>

#include <atomic>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

enum class TraceTerminal : std::uint8_t {
  SampledFailure,
  BlitFailure,
  ResolveQueueFailure,
  ResolveCompletion,
};

[[nodiscard]] constexpr std::uint64_t
TraceCommandSubmits(const std::uint64_t completed,
                    const TraceTerminal terminal) noexcept {
  return terminal == TraceTerminal::ResolveCompletion
             ? ::rund::detail::counter::SaturatingAdd(completed, 1u)
             : completed;
}

static_assert(TraceCommandSubmits(1u, TraceTerminal::SampledFailure) == 1u);
static_assert(TraceCommandSubmits(1u, TraceTerminal::BlitFailure) == 1u);
static_assert(TraceCommandSubmits(1u, TraceTerminal::ResolveQueueFailure) ==
              1u);
static_assert(TraceCommandSubmits(1u, TraceTerminal::ResolveCompletion) == 2u);

[[nodiscard]] std::atomic_ref<MetalAdapter *>
TraceSubmitAdapter(MetalDispatchTrace &trace) noexcept {
  return std::atomic_ref<MetalAdapter *>{trace.submit_adapter};
}

static_assert(std::atomic_ref<MetalAdapter *>::is_always_lock_free);

void FinishTraceSubmission(MetalDispatchTrace &trace, KernelResult result,
                           const TraceTerminal terminal) noexcept {
  auto submit_adapter = TraceSubmitAdapter(trace);
  // Queue publication releases through submit_adapter; this acquire makes the
  // callback's completion/user and physical counter fields visible.
  static_cast<void>(submit_adapter.load(std::memory_order_acquire));
  KernelCompletion const completion = trace.submit_completion;
  void *const user = trace.submit_user;
  result.stats.run.work.command_submit_count =
      TraceCommandSubmits(result.stats.run.work.command_submit_count, terminal);
  trace.submit_completion = nullptr;
  trace.submit_user = nullptr;
  submit_adapter.store(nullptr, std::memory_order_release);
  if (completion != nullptr) {
    completion(user, result);
  }
}

void CompleteTraceResolve(void *const raw, KernelResult result) noexcept {
  auto *const trace = static_cast<MetalDispatchTrace *>(raw);
  if (trace == nullptr) {
    return;
  }
  FinishTraceSubmission(*trace, result, TraceTerminal::ResolveCompletion);
}

void CompleteTraceSamples(void *const raw, KernelResult result) noexcept {
  auto *const trace = static_cast<MetalDispatchTrace *>(raw);
  if (trace == nullptr) {
    return;
  }
  MetalAdapter *const adapter =
      TraceSubmitAdapter(*trace).load(std::memory_order_acquire);
  if (!result.check.ok || adapter == nullptr) {
    FinishTraceSubmission(*trace, result, TraceTerminal::SampledFailure);
    return;
  }
  id<MTLCommandQueue> const queue =
      (__bridge id<MTLCommandQueue>)adapter->queue.get();
  id<MTLCommandBuffer> const command =
      queue == nil ? nil : [queue commandBufferWithUnretainedReferences];
  id<MTLBlitCommandEncoder> const blit =
      command == nil ? nil : [command blitCommandEncoder];
  if (blit == nil) {
    result.check = rund::AccelCheck{false, "accel_metal_command_unavailable"};
    FinishTraceSubmission(*trace, result, TraceTerminal::BlitFailure);
    return;
  }
  [blit resolveCounters:trace->samples
                inRange:NSMakeRange(0u, trace->sample_count)
      destinationBuffer:trace->values
      destinationOffset:0u];
  [blit endEncoding];
  if (adapter->fault_trace_resolve_device_lost_once.exchange(
          false, std::memory_order_relaxed)) {
    adapter->fault_device_lost_once.store(true, std::memory_order_relaxed);
  }
  const rund::AccelCheck queued = QueueCommand(
      *adapter, (__bridge void *)command, CompleteTraceResolve, trace, false);
  if (!queued.ok) {
    result.check = queued;
    FinishTraceSubmission(*trace, result, TraceTerminal::ResolveQueueFailure);
  }
}

} // namespace

rund::AccelCheck QueueMetalDispatchTrace(
    MetalAdapter &adapter, id<MTLCommandBuffer> const sampled_command,
    MetalDispatchTrace &trace, const KernelCompletion completion,
    void *const user) noexcept {
  auto submit_adapter = TraceSubmitAdapter(trace);
  if (sampled_command == nil || !trace.available() || trace.failed ||
      completion == nullptr ||
      submit_adapter.load(std::memory_order_acquire) != nullptr) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  trace.submit_completion = completion;
  trace.submit_user = user;
  submit_adapter.store(&adapter, std::memory_order_release);
  const rund::AccelCheck queued =
      QueueCommand(adapter, (__bridge void *)sampled_command,
                   CompleteTraceSamples, &trace, false);
  if (!queued.ok) {
    trace.submit_completion = nullptr;
    trace.submit_user = nullptr;
    submit_adapter.store(nullptr, std::memory_order_release);
  }
  return queued;
}

#endif

} // namespace rund::node::accel::detail
