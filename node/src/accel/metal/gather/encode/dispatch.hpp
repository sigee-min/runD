#pragma once

#include "prepare.hpp"
#include "../../../kernel/preparation.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalGatherDispatch(const MetalGatherCommandState& state,
                                      const GatherParams& params) {
  const auto &gather = *state.gather;
  [state.encoder setComputePipelineState:state.control_pipeline];
  [state.encoder setBuffer:state.logical_count
                    offset:gather.plan.count_source ==
                                   rund::kernel::ComputeCountSource::Descriptor
                               ? 0u
                               : static_cast<NSUInteger>(
                                     gather.logical_count.ref.offset_bytes)
                   atIndex:0u];
  [state.encoder setBuffer:state.indices
                    offset:static_cast<NSUInteger>(gather.indices.ref.offset_bytes)
                   atIndex:1u];
  [state.encoder setBuffer:state.status offset:0u atIndex:2u];
  [state.encoder setBuffer:state.indirect offset:0u atIndex:3u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:4u];
  [state.encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(kGatherThreadgroupSize, 1u,
                                                  1u)];
  [state.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

  [state.encoder setComputePipelineState:state.gather_pipeline];
  [state.encoder setBuffer:state.values
                    offset:static_cast<NSUInteger>(gather.values.ref.offset_bytes)
                   atIndex:0u];
  [state.encoder setBuffer:state.indices
                    offset:static_cast<NSUInteger>(gather.indices.ref.offset_bytes)
                   atIndex:1u];
  [state.encoder setBuffer:state.output
                    offset:static_cast<NSUInteger>(gather.output.ref.offset_bytes)
                   atIndex:2u];
  [state.encoder setBuffer:state.indirect offset:0u atIndex:3u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:4u];
  const MTLSize threads = MTLSizeMake(kGatherThreadgroupSize, 1u, 1u);
  if (IsPipelinePrivatePreparation(CurrentKernelPreparationMode())) {
    const std::uint64_t groups =
        gather.plan.element_count / kGatherThreadgroupSize +
        (gather.plan.element_count % kGatherThreadgroupSize != 0u ? 1u : 0u);
    [state.encoder dispatchThreadgroups:MTLSizeMake(
                                            static_cast<NSUInteger>(groups),
                                            1u, 1u)
                    threadsPerThreadgroup:threads];
  } else {
    [state.encoder dispatchThreadgroupsWithIndirectBuffer:state.indirect
                                     indirectBufferOffset:0u
                                    threadsPerThreadgroup:threads];
  }
}
#endif

}  // namespace rund::node::accel::detail
