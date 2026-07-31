#pragma once

#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalFlagScanBlock(const MetalFlagScanEncodeState &state) {
  [state.encoder setComputePipelineState:state.block];
  [state.encoder setBuffer:state.flags offset:state.flags_offset atIndex:0u];
  [state.encoder setBuffer:state.output offset:state.output_offset atIndex:1u];
  [state.encoder setBuffer:state.totals offset:state.totals_offset atIndex:2u];
  [state.encoder setBuffer:state.status offset:0u atIndex:3u];
  [state.encoder setBytes:&state.element_count
                   length:sizeof(state.element_count)
                  atIndex:4u];
  [state.encoder setBytes:&state.block_size
                   length:sizeof(state.block_size)
                  atIndex:5u];
  [state.encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(
                                                      state.block_count),
                                                  1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(state.block_threads, 1u, 1u)];
}
#endif

} // namespace rund::node::accel::detail
