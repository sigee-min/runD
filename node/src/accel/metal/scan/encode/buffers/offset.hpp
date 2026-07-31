#pragma once

#include "prefix.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void EncodeMetalScanOffset(
    void *const output_buffer, void *const totals_buffer,
    void *const status_buffer, void *const input_buffer,
    void *const logical_count_buffer, const rund::kernel::u32 count_words,
    const rund::kernel::u32 signed_domain, const rund::kernel::u32 inclusive,
    const MetalScanEncodeState &state, const rund::kernel::u64 input_offset,
    const rund::kernel::u64 output_offset,
    const rund::kernel::u64 logical_count_offset,
    const rund::kernel::u64 totals_offset) {
  [state.encoder setComputePipelineState:state.offset];
  [state.encoder setBuffer:(__bridge id<MTLBuffer>)output_buffer
                    offset:static_cast<NSUInteger>(output_offset)
                   atIndex:0u];
  [state.encoder setBuffer:(__bridge id<MTLBuffer>)totals_buffer
                    offset:static_cast<NSUInteger>(totals_offset)
                   atIndex:1u];
  [state.encoder setBuffer:(__bridge id<MTLBuffer>)status_buffer
                    offset:0u
                   atIndex:2u];
  [state.encoder setBytes:&state.element_count
                   length:sizeof(state.element_count)
                  atIndex:3u];
  [state.encoder setBytes:&state.block_size
                   length:sizeof(state.block_size)
                  atIndex:4u];
  if (logical_count_buffer != nullptr) {
    [state.encoder setBuffer:(__bridge id<MTLBuffer>)logical_count_buffer
                      offset:static_cast<NSUInteger>(logical_count_offset)
                     atIndex:5u];
  }
  [state.encoder setBytes:&count_words length:sizeof(count_words) atIndex:6u];
  [state.encoder setBytes:&signed_domain
                   length:sizeof(signed_domain)
                  atIndex:7u];
  [state.encoder setBuffer:(__bridge id<MTLBuffer>)input_buffer
                    offset:static_cast<NSUInteger>(input_offset)
                   atIndex:8u];
  [state.encoder setBytes:&inclusive length:sizeof(inclusive) atIndex:9u];
  [state.encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(
                                                      state.block_count),
                                                  1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(state.block_threads, 1u, 1u)];
}
#endif

} // namespace rund::node::accel::detail
