#pragma once

#include <accel/check.hpp>

#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
EncodeMetalCompactElementPath(MetalAdapter &adapter,
                              void *const command_encoder,
                              const MetalCompactEncodeState &state) {
  const rund::AccelCheck scan = EncodeMetalScanU32FlagBuffers(
      adapter, state.compact->scan_desc, state.compact->scan_plan,
      (__bridge void *)state.flags, state.flags_offset,
      (__bridge void *)state.offsets, state.compact->offsets.offset,
      state.compact->scan_totals.buffer.get(),
      state.compact->scan_totals.offset,
      state.compact->scan_status.buffer.get(), command_encoder,
      state.compact->scan_block, state.compact->scan_prefix,
      state.compact->scan_offset, false);
  if (!scan.ok) {
    return scan;
  }
  const MTLSize grid = MTLSizeMake(
      static_cast<NSUInteger>(state.compact->plan.element_count), 1u, 1u);
  const MTLSize threads = MTLSizeMake(block::MetalCompact, 1u, 1u);
  [state.encoder setComputePipelineState:state.scatter];
  [state.encoder setBuffer:state.flags offset:state.flags_offset atIndex:0u];
  [state.encoder setBuffer:state.offsets
                    offset:state.offsets_offset
                   atIndex:1u];
  [state.encoder setBuffer:state.output offset:state.output_offset atIndex:2u];
  [state.encoder
      setBuffer:(__bridge id<MTLBuffer>)state.compact->scan_totals.buffer.get()
         offset:state.scan_totals_offset
        atIndex:3u];
  [state.encoder setBytes:&state.params length:sizeof(state.params) atIndex:4u];
  [state.encoder dispatchThreads:grid threadsPerThreadgroup:threads];
  if (state.status_required) {
    [state.encoder setComputePipelineState:state.status_pipeline];
    [state.encoder setBuffer:state.flags offset:state.flags_offset atIndex:0u];
    [state.encoder setBuffer:state.offsets
                      offset:state.offsets_offset
                     atIndex:1u];
    [state.encoder setBuffer:(__bridge id<MTLBuffer>)
                                 state.compact->scan_totals.buffer.get()
                      offset:state.scan_totals_offset
                     atIndex:2u];
    [state.encoder setBuffer:state.status offset:0u atIndex:3u];
    [state.encoder setBytes:&state.params
                     length:sizeof(state.params)
                    atIndex:4u];
    [state.encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
