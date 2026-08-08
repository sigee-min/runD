#pragma once

#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalStencilDispatch(const MetalStencilCommandState &state,
                                       const std::uint32_t stage_index,
                                       const StencilParams &params) {
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)state.stencil
          ->pipelines[stage_index]
          .get();
  const RangeAggregateStagePlan stage = state.stencil->range.stage(stage_index);
  [state.encoder setComputePipelineState:pipeline];
  [state.encoder
      setBuffer:state.input
         offset:static_cast<NSUInteger>(state.stencil->input.ref.offset_bytes)
        atIndex:0u];
  [state.encoder
      setBuffer:state.output
         offset:static_cast<NSUInteger>(state.stencil->output.ref.offset_bytes)
        atIndex:1u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:2u];
  if (StencilRangeUsesGlobalScratch(state.stencil->range)) {
    const MetalRuntimeBuffer *scratch0 = nullptr;
    const MetalRuntimeBuffer *scratch1 = nullptr;
    if (MetalStencilStageScratchBindings(*state.stencil, stage_index, scratch0,
                                         scratch1) &&
        scratch0 != nullptr && scratch1 != nullptr) {
      [state.encoder setBuffer:(__bridge id<MTLBuffer>)scratch0->buffer.get()
                        offset:static_cast<NSUInteger>(scratch0->offset)
                       atIndex:3u];
      [state.encoder setBuffer:(__bridge id<MTLBuffer>)scratch1->buffer.get()
                        offset:static_cast<NSUInteger>(scratch1->offset)
                       atIndex:4u];
    }
  }
  [state.encoder
       dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(stage.groups),
                                        1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(state.stencil->shape.width(), 1u, 1u)];
}
#endif

} // namespace rund::node::accel::detail
