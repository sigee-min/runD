#include "model.hpp"

#include "../../../kernel/preparation.hpp"
#include "../../../segmented/reduce/metal.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <algorithm>
#include <cstring>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] id<MTLBuffer> Buffer(const std::shared_ptr<void> &buffer) {
  return (__bridge id<MTLBuffer>)buffer.get();
}

} // namespace
#endif

rund::AccelCheck
EncodeMetalSegmentedReduce(MetalAdapter &adapter,
                           const std::shared_ptr<void> &resources,
                           void *const command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const state =
      static_cast<MetalSegmentedReduceResources *>(resources.get());
  id<MTLComputeCommandEncoder> encoder =
      (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  void *const status =
      state == nullptr ? nullptr : MetalBufferContents(state->status);
  if (state == nullptr || state->adapter != &adapter || encoder == nil ||
      status == nullptr || state->segments_per_group == 0u ||
      state->pipelines.classify == nullptr ||
      state->pipelines.prefix == nullptr ||
      state->pipelines.scatter == nullptr ||
      state->pipelines.reduce == nullptr) {
    return {false, "compute_segmented_reduce_invalid"};
  }
  std::memset(status, 0, sizeof(rund::kernel::u32));
  id<MTLComputePipelineState> classify =
      (__bridge id<MTLComputePipelineState>)state->pipelines.classify.get();
  id<MTLComputePipelineState> prefix =
      (__bridge id<MTLComputePipelineState>)state->pipelines.prefix.get();
  id<MTLComputePipelineState> scatter =
      (__bridge id<MTLComputePipelineState>)state->pipelines.scatter.get();
  id<MTLComputePipelineState> reduce =
      (__bridge id<MTLComputePipelineState>)state->pipelines.reduce.get();
  const SegmentedReduceLayout layout =
      SegmentedReduceLayoutFor(state->plan.element_count);
  const MetalSegmentedReduceParams params{
      state->plan.element_count, layout.block_count, state->segments_per_group};
  const MTLSize groups =
      MTLSizeMake(static_cast<NSUInteger>(layout.index_groups), 1u, 1u);
  const MTLSize threads =
      MTLSizeMake(static_cast<NSUInteger>(kSegmentedIndexWidth), 1u, 1u);
  [encoder setComputePipelineState:classify];
  [encoder setBuffer:Buffer(state->heads.device_buffer)
              offset:static_cast<NSUInteger>(state->heads.ref.offset_bytes)
             atIndex:0];
  [encoder setBuffer:Buffer(state->block_counts.buffer)
              offset:static_cast<NSUInteger>(state->block_counts.offset)
             atIndex:1];
  [encoder setBuffer:Buffer(state->status.buffer) offset:0 atIndex:4];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [encoder setComputePipelineState:prefix];
  [encoder setBuffer:Buffer(state->block_counts.buffer)
              offset:static_cast<NSUInteger>(state->block_counts.offset)
             atIndex:0];
  [encoder setBuffer:Buffer(state->block_offsets.buffer)
              offset:static_cast<NSUInteger>(state->block_offsets.offset)
             atIndex:1];
  [encoder setBuffer:Buffer(state->segment_count.buffer) offset:0 atIndex:2];
  [encoder setBuffer:Buffer(state->dispatch_args.buffer) offset:0 atIndex:3];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  [encoder dispatchThreads:threads threadsPerThreadgroup:threads];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [encoder setComputePipelineState:scatter];
  [encoder setBuffer:Buffer(state->heads.device_buffer)
              offset:static_cast<NSUInteger>(state->heads.ref.offset_bytes)
             atIndex:0];
  [encoder setBuffer:Buffer(state->block_offsets.buffer)
              offset:static_cast<NSUInteger>(state->block_offsets.offset)
             atIndex:1];
  [encoder setBuffer:Buffer(state->segment_starts.buffer)
              offset:static_cast<NSUInteger>(state->segment_starts.offset)
             atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [encoder setComputePipelineState:reduce];
  [encoder setBuffer:Buffer(state->input.device_buffer)
              offset:static_cast<NSUInteger>(state->input.ref.offset_bytes)
             atIndex:0];
  [encoder setBuffer:Buffer(state->segment_starts.buffer)
              offset:static_cast<NSUInteger>(state->segment_starts.offset)
             atIndex:1];
  [encoder setBuffer:Buffer(state->segment_count.buffer) offset:0 atIndex:2];
  [encoder setBuffer:Buffer(state->output.device_buffer)
              offset:static_cast<NSUInteger>(state->output.ref.offset_bytes)
             atIndex:3];
  [encoder setBuffer:Buffer(state->status.buffer) offset:0 atIndex:4];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  if (IsPipelinePrivatePreparation(CurrentKernelPreparationMode())) {
    const std::uint64_t maximum =
        state->plan.element_count / state->segments_per_group +
        (state->plan.element_count % state->segments_per_group != 0u ? 1u : 0u);
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>(
                                          std::min<std::uint64_t>(
                                              maximum, kSegmentedMaxGroups)),
                                      1u, 1u)
            threadsPerThreadgroup:threads];
  } else {
    [encoder
        dispatchThreadgroupsWithIndirectBuffer:Buffer(
                                                   state->dispatch_args.buffer)
                          indirectBufferOffset:0u
                         threadsPerThreadgroup:threads];
  }
  return {true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return {false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
