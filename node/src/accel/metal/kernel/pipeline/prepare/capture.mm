#include "../build.hpp"

#include "../aggregate/prepare.hpp"

#include <algorithm>
#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Capture() {
  if (finished) {
    return rund::AccelCheck{true, "ok"};
  }
  encoder = [[RUNDMetalPipelineCapture alloc] initWithCapture:&captured];
  captured.guard_zero = pipeline->guard_zero;
  captured.guard_states = pipeline->states;
  captured.guard_state_count = pipeline->state_count;
  if (aggregate_selected) {
    captured.unguarded = true;
    reset_command_count = 0u;
    id<MTLBuffer> const step_control =
        profile_steps ? pipeline->step_control : pipeline->control;
    const rund::AccelCheck encoded = EncodeMetalNestedAggregate(
        native_aggregate, pipeline->control, step_control, encoder);
    if (encoded.ok && captured.commands.size() == 2u) {
      captured.commands.back().control = true;
    }
    return encoded;
  }
  MetalPipelineStatusParams open = status_params;
  open.phase = 0u;
  [encoder setComputePipelineState:complete];
  [encoder setBuffer:pipeline->control offset:0u atIndex:0u];
  [encoder setBytes:&open length:sizeof(open) atIndex:1u];
  if (profile_steps) {
    [encoder setBuffer:pipeline->step_control offset:0u atIndex:2u];
  }
  id<MTLBuffer> const states =
      pipeline->states == nil ? pipeline->control : pipeline->states;
  [encoder setBuffer:states offset:0u atIndex:3u];
  const NSUInteger count = std::max<NSUInteger>(pipeline->state_count, 1u);
  const NSUInteger width =
      std::min(count, [complete maxTotalThreadsPerThreadgroup]);
  [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
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
