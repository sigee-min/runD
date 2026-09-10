#include "local.hpp"

#include "../../pipeline/named.hpp"
#include "../trace/encoder.hpp"
#include "../trace/replay.hpp"
#include "../trace/submit.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
rund::AccelCheck SubmitMetalResources(const rund::AccelDevice &pick,
                                      const std::shared_ptr<void> &prepared,
                                      const KernelCompletion completion,
                                      void *const user,
                                      PreparedMemoryMeter *const memory,
                                      const KernelTiming timing) noexcept {
  @autoreleasepool {
    auto *const resources = static_cast<MetalKernelResources *>(prepared.get());
    if (resources == nullptr || resources->size() == 0u ||
        completion == nullptr) {
      return rund::AccelCheck{false, "accel_metal_unavailable"};
    }
    MetalKernelContext context{};
    const rund::AccelCheck valid = ValidateMetalKernelContext(pick, context);
    if (!valid.ok) {
      return valid;
    }
    submission::State<MetalKernelResources> &state = resources->submission;
    {
      std::lock_guard lock{state.mutex};
      if (state.active()) {
        return rund::AccelCheck{false, "compute_job_busy"};
      }
      if (timing == KernelTiming::Dispatch) {
        const rund::AccelCheck traced = EnsureMetalPreparedDispatchTrace(
            (__bridge id<MTLDevice>)context.adapter->device.get(), *resources,
            memory);
        if (!traced.ok) {
          return traced;
        }
      }
      state.owner = resources;
      state.completion = completion;
      state.user = user;
    }
    CommandRun command{};
    id trace_encoder = nil;
    rund::AccelCheck encoded{};
    if (timing == KernelTiming::Dispatch &&
        resources->trace.sampling == MetalTraceSampling::StageBoundary) {
      encoded = EncodeMetalPreparedStageTrace(
          *context.adapter, *resources,
          (__bridge id<MTLDevice>)context.adapter->device.get(), command);
    } else {
      const rund::AccelCheck ready =
          OpenCommand<ResourceRefs::Borrowed>(*context.adapter, command);
      if (!ready.ok) {
        submission::Cancel(state);
        return ready;
      }
      if (timing == KernelTiming::Dispatch) {
        BeginMetalDispatchTrace(
            (__bridge id<MTLDevice>)context.adapter->device.get(),
            resources->trace);
        trace_encoder = resources->trace_encoder;
        BindMetalDispatchTraceEncoder(trace_encoder, command.encoder,
                                      resources->trace);
        command.encoder = (id<MTLComputeCommandEncoder>)trace_encoder;
      }
      encoded = EncodeMetalSteps(*context.adapter, *resources, command);
      ClearMetalDispatchTraceEncoder(trace_encoder);
      if (encoded.ok && timing == KernelTiming::Dispatch &&
          !SealMetalDispatchTrace(command.buffer, resources->trace)) {
        encoded =
            rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
      }
    }
    if (!encoded.ok) {
      submission::Cancel(state);
      return encoded;
    }
    const bool trace = timing == KernelTiming::Dispatch;
    rund::AccelCheck submitted{};
    {
      // Queue publication and terminal Take share this gate, making every
      // encode-side host write visible before completion observes resources.
      std::lock_guard lock{state.mutex};
      submitted =
          trace
              ? QueueMetalDispatchTrace(*context.adapter, command.buffer,
                                        resources->trace,
                                        CompleteMetalPreparedTrace, &state)
              : QueueCommand(*context.adapter, (__bridge void *)command.buffer,
                             CompleteMetalPrepared, &state,
                             timing == KernelTiming::Submission);
    }
    if (!submitted.ok) {
      submission::Cancel(state);
    }
    return submitted;
  }
}
#else
rund::AccelCheck SubmitMetalResources(const rund::AccelDevice &,
                                      const std::shared_ptr<void> &,
                                      KernelCompletion, void *,
                                      PreparedMemoryMeter *,
                                      KernelTiming) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}
#endif

} // namespace rund::node::accel::detail
