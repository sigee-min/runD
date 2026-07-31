#pragma once

#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalStencilDispatch(const MetalStencilCommandState &state,
                                       const StencilParams &params) {
  [state.encoder setComputePipelineState:state.pipeline];
  [state.encoder
      setBuffer:state.input
         offset:static_cast<NSUInteger>(state.stencil->input.ref.offset_bytes)
        atIndex:0u];
  [state.encoder
      setBuffer:state.output
         offset:static_cast<NSUInteger>(state.stencil->output.ref.offset_bytes)
        atIndex:1u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:2u];
  [state.encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(
                                                 params.element_count),
                                             1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(kStencilThreadgroupSize, 1u, 1u)];
}
#endif

} // namespace rund::node::accel::detail
