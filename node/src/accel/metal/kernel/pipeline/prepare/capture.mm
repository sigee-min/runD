#include "../build.hpp"

#include <algorithm>
#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Capture() {
  if (finished) {
    return rund::AccelCheck{true, "ok"};
  }
  try {
    window_params.reserve(native_windows.size());
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  encoder = [[RUNDMetalPipelineCapture alloc] initWithCapture:&captured];
  MetalPipelineStatusParams open = status_params;
  open.phase = 0u;
  [encoder setComputePipelineState:complete];
  [encoder setBuffer:pipeline->control offset:0u atIndex:0u];
  [encoder setBytes:&open length:sizeof(open) atIndex:1u];
  if (profile_steps) {
    [encoder setBuffer:pipeline->step_control offset:0u atIndex:2u];
  }
  [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
  captured.commands.back().control = true;
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  if (private_raw_count != 0u) {
    [encoder setComputePipelineState:reset];
    [encoder setBuffer:pipeline->raw_status offset:0u atIndex:0u];
    [encoder setBytes:status_resets.data()
               length:status_resets.size() * sizeof(MetalPipelineResetMeta)
              atIndex:1u];
    [encoder setBytes:&status_params length:sizeof(status_params) atIndex:2u];
    const NSUInteger reset_threads = private_raw_count;
    const NSUInteger reset_width =
        std::min(reset_threads, [reset maxTotalThreadsPerThreadgroup]);
    [encoder dispatchThreads:MTLSizeMake(reset_threads, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(reset_width, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  return EncodePrograms();
}

#endif

} // namespace rund::node::accel::detail
