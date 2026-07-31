#pragma once

#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalScatterDispatch(const MetalScatterCommandState &state,
                                       const ScatterParams &params) {
  [state.encoder setComputePipelineState:state.pipeline];
  [state.encoder
      setBuffer:state.values
         offset:static_cast<NSUInteger>(state.scatter->values.ref.offset_bytes)
        atIndex:0u];
  [state.encoder
      setBuffer:state.indices
         offset:static_cast<NSUInteger>(state.scatter->indices.ref.offset_bytes)
        atIndex:1u];
  [state.encoder
      setBuffer:state.output
         offset:static_cast<NSUInteger>(state.scatter->output.ref.offset_bytes)
        atIndex:2u];
  [state.encoder setBuffer:state.status offset:0u atIndex:3u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:4u];
  [state.encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(
                                                 params.element_count),
                                             1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(kScatterThreadgroupSize, 1u, 1u)];
}
#endif

} // namespace rund::node::accel::detail
