#include "model.hpp"

#include "../../../scatter/reduce/model.hpp"

#include <cstdint>
#include <memory>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

rund::AccelCheck
EncodeMetalScatterReduce(MetalAdapter &adapter,
                         const std::shared_ptr<void> &resources,
                         void *const command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const state =
      static_cast<MetalScatterReduceResources *>(resources.get());
  id<MTLComputeCommandEncoder> encoder =
      (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  if (state == nullptr || state->adapter != &adapter || encoder == nil) {
    return {false, "compute_scatter_reduce_buffer_invalid"};
  }
  id<MTLComputePipelineState> control =
      (__bridge id<MTLComputePipelineState>)state->control_pipeline.get();
  id<MTLComputePipelineState> fold =
      (__bridge id<MTLComputePipelineState>)state->fold_pipeline.get();
  id<MTLComputePipelineState> init =
      (__bridge id<MTLComputePipelineState>)state->init_pipeline.get();
  id<MTLBuffer> values =
      (__bridge id<MTLBuffer>)state->values.device_buffer.get();
  id<MTLBuffer> indices =
      (__bridge id<MTLBuffer>)state->indices.device_buffer.get();
  id<MTLBuffer> count =
      state->plan.count_source == rund::kernel::ComputeCountSource::Descriptor
          ? values
          : (__bridge id<MTLBuffer>)state->count.device_buffer.get();
  id<MTLBuffer> output =
      (__bridge id<MTLBuffer>)state->output.device_buffer.get();
  id<MTLBuffer> status = (__bridge id<MTLBuffer>)state->status.buffer.get();
  id<MTLBuffer> indirect = (__bridge id<MTLBuffer>)state->indirect.buffer.get();
  id<MTLBuffer> counts = (__bridge id<MTLBuffer>)state->counts.buffer.get();
  if (control == nil || init == nil || fold == nil || values == nil ||
      indices == nil || count == nil || output == nil || status == nil ||
      indirect == nil || counts == nil) {
    return {false, "accel_metal_command_unavailable"};
  }
  const ScatterReduceParams params{
      state->plan.element_count, state->plan.output_count,
      static_cast<std::uint32_t>(state->plan.count_source), 0u};
  const auto bind = [&](id<MTLComputePipelineState> pipeline) {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:values
                offset:static_cast<NSUInteger>(state->values.ref.offset_bytes)
               atIndex:0u];
    [encoder setBuffer:indices
                offset:static_cast<NSUInteger>(state->indices.ref.offset_bytes)
               atIndex:1u];
    [encoder
        setBuffer:count
           offset:state->plan.count_source ==
                          rund::kernel::ComputeCountSource::Descriptor
                      ? 0u
                      : static_cast<NSUInteger>(state->count.ref.offset_bytes)
          atIndex:2u];
    [encoder setBuffer:output
                offset:static_cast<NSUInteger>(state->output.ref.offset_bytes)
               atIndex:3u];
    [encoder setBuffer:status offset:0u atIndex:4u];
    [encoder setBuffer:indirect offset:0u atIndex:5u];
    [encoder setBytes:&params length:sizeof(params) atIndex:6u];
    [encoder setBuffer:counts
                offset:static_cast<NSUInteger>(state->counts.offset)
               atIndex:7u];
  };
  bind(control);
  [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(kScatterReduceWidth, 1u, 1u)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  bind(init);
  [encoder
      dispatchThreadgroupsWithIndirectBuffer:indirect
                        indirectBufferOffset:0u
                       threadsPerThreadgroup:MTLSizeMake(kScatterReduceWidth,
                                                         1u, 1u)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  bind(fold);
  [encoder
      dispatchThreadgroupsWithIndirectBuffer:indirect
                        indirectBufferOffset:3u * sizeof(std::uint32_t)
                       threadsPerThreadgroup:
                           MTLSizeMake(rund::kernel::ScatterReduceFoldParallel(
                                           state->plan)
                                           ? kScatterReduceWidth
                                           : 1u,
                                       1u, 1u)];
  return {true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return {false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
