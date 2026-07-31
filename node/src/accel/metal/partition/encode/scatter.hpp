#pragma once

#include "scan.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalPartitionScatter(const MetalPartitionCommandState &state,
                                        const PartitionParams &params) {
  const NSUInteger width = std::max<NSUInteger>(
      1u, std::min<NSUInteger>(kPartitionThreadgroupSize,
                               [state.scatter maxTotalThreadsPerThreadgroup]));
  [state.encoder setComputePipelineState:state.scatter];
  [state.encoder setBuffer:state.flags offset:state.flags_offset atIndex:0u];
  [state.encoder setBuffer:state.values offset:state.values_offset atIndex:1u];
  [state.encoder setBuffer:state.output offset:state.output_offset atIndex:2u];
  [state.encoder setBuffer:state.false_offsets
                    offset:state.false_offsets_offset
                   atIndex:3u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:4u];
  [state.encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(
                                                 params.element_count),
                                             1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}
#endif

} // namespace rund::node::accel::detail
