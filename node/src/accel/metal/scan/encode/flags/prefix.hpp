#pragma once

#include "block.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalFlagScanPrefix(const MetalFlagScanEncodeState& state) {
  [state.encoder setComputePipelineState:state.prefix];
  [state.encoder setBuffer:state.totals offset:state.totals_offset atIndex:0u];
  [state.encoder setBytes:&state.block_count
                   length:sizeof(state.block_count)
                  atIndex:1u];
  [state.encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
                     threadsPerThreadgroup:MTLSizeMake(state.prefix_threads,
                                                       1u, 1u)];
}
#endif

}  // namespace rund::node::accel::detail
