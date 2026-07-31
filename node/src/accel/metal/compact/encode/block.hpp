#pragma once

#include <accel/check.hpp>

#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
EncodeMetalCompactBlockPath(MetalAdapter &adapter, void *const command_encoder,
                            const MetalCompactEncodeState &state) {
  const MTLSize groups =
      MTLSizeMake(static_cast<NSUInteger>(state.compact->block_count), 1u, 1u);
  const MTLSize threads = MTLSizeMake(block::MetalCompact, 1u, 1u);
  [state.encoder setComputePipelineState:state.count_blocks];
  [state.encoder setBuffer:state.flags offset:state.flags_offset atIndex:0u];
  [state.encoder setBuffer:state.block_counts
                    offset:state.block_counts_offset
                   atIndex:1u];
  [state.encoder setBuffer:state.flag_bits
                    offset:state.flag_bits_offset
                   atIndex:2u];
  [state.encoder setBytes:&state.params length:sizeof(state.params) atIndex:3u];
  [state.encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  const rund::AccelCheck scan = EncodeMetalScanDeferredOffsetBuffers(
      adapter, state.compact->scan_desc, state.compact->scan_plan,
      (__bridge void *)state.block_counts, (__bridge void *)state.block_offsets,
      state.compact->scan_totals.buffer.get(),
      state.compact->scan_status.buffer.get(), command_encoder,
      state.compact->block_counts.offset,
      state.compact->block_offsets.offset,
      state.compact->scan_totals.offset);
  if (!scan.ok) {
    return scan;
  }
  [state.encoder setComputePipelineState:state.scatter_blocks];
  [state.encoder setBuffer:state.flag_bits
                    offset:state.flag_bits_offset
                   atIndex:0u];
  [state.encoder setBuffer:state.block_offsets
                    offset:state.block_offsets_offset
                   atIndex:1u];
  [state.encoder setBuffer:state.output offset:state.output_offset atIndex:2u];
  [state.encoder
      setBuffer:(__bridge id<MTLBuffer>)state.compact->scan_totals.buffer.get()
         offset:state.scan_totals_offset
        atIndex:3u];
  [state.encoder setBytes:&state.params length:sizeof(state.params) atIndex:4u];
  [state.encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
