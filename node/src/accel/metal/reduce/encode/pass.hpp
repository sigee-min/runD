#pragma once

#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalReducePass(const MetalReduceCommandState &state,
                                  const ReducePassParams &params,
                                  const bool read_partial,
                                  const rund::kernel::u64 next) {
  const auto block_size =
      static_cast<NSUInteger>(state.reduce->plan.block_size);
  [state.encoder setBuffer:(read_partial ? state.partial : state.input)
                    offset:(read_partial ? state.partial_offset
                                         : state.input_offset)
                   atIndex:0u];
  [state.encoder setBuffer:state.partial
                    offset:state.partial_offset
                   atIndex:1u];
  [state.encoder setBuffer:state.output
                    offset:state.output_offset
                   atIndex:2u];
  [state.encoder setBuffer:state.status offset:0u atIndex:3u];
  [state.encoder setBytes:&params length:sizeof(params) atIndex:4u];
  if (state.logical_count != nil) {
    [state.encoder setBuffer:state.logical_count
                      offset:state.logical_count_offset
                     atIndex:5u];
  }
  [state.encoder
       dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(next), 1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(block_size, 1u, 1u)];
}
#endif

} // namespace rund::node::accel::detail
