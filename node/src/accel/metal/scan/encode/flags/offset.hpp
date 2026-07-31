#pragma once

#include "prefix.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalFlagScanOffset(const MetalFlagScanEncodeState &state) {
  [state.encoder setComputePipelineState:state.offset];
  [state.encoder setBuffer:state.output offset:state.output_offset atIndex:0u];
  [state.encoder setBuffer:state.totals offset:state.totals_offset atIndex:1u];
  [state.encoder setBuffer:state.status offset:0u atIndex:2u];
  [state.encoder setBytes:&state.element_count
                   length:sizeof(state.element_count)
                  atIndex:3u];
  [state.encoder setBytes:&state.block_size
                   length:sizeof(state.block_size)
                  atIndex:4u];
  const rund::kernel::u32 count_words = 0u;
  [state.encoder setBytes:&count_words length:sizeof(count_words) atIndex:6u];
  const rund::kernel::u32 signed_domain = 0u;
  [state.encoder setBytes:&signed_domain
                   length:sizeof(signed_domain)
                  atIndex:7u];
  [state.encoder setBuffer:state.flags offset:state.flags_offset atIndex:8u];
  const rund::kernel::u32 inclusive = 0u;
  [state.encoder setBytes:&inclusive length:sizeof(inclusive) atIndex:9u];
  [state.encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(
                                                      state.block_count),
                                                  1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(state.block_threads, 1u, 1u)];
}
#endif

} // namespace rund::node::accel::detail
