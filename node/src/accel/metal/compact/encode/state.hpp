#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalCompactEncodeState {
  MetalCompactEncodeResources *compact = nullptr;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLComputePipelineState> scatter = nil;
  id<MTLComputePipelineState> count_blocks = nil;
  id<MTLComputePipelineState> scatter_blocks = nil;
  id<MTLComputePipelineState> status_pipeline = nil;
  id<MTLBuffer> flags = nil;
  id<MTLBuffer> output = nil;
  id<MTLBuffer> offsets = nil;
  id<MTLBuffer> flag_bits = nil;
  id<MTLBuffer> block_counts = nil;
  id<MTLBuffer> block_offsets = nil;
  id<MTLBuffer> status = nil;
  NSUInteger flags_offset = 0u;
  NSUInteger output_offset = 0u;
  NSUInteger offsets_offset = 0u;
  NSUInteger flag_bits_offset = 0u;
  NSUInteger block_counts_offset = 0u;
  NSUInteger block_offsets_offset = 0u;
  NSUInteger scan_totals_offset = 0u;
  CompactParams params{};
  bool status_required = false;
};
#endif

} // namespace rund::node::accel::detail
