#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalSortPassBuffers {
  id<MTLBuffer> source_keys = nil;
  id<MTLBuffer> source_values = nil;
  id<MTLBuffer> target_keys = nil;
  id<MTLBuffer> target_values = nil;
  NSUInteger source_keys_offset = 0u;
  NSUInteger source_values_offset = 0u;
  NSUInteger target_keys_offset = 0u;
  NSUInteger target_values_offset = 0u;
};

struct MetalSortEncodeState {
  MetalSortEncodeResources *sort = nullptr;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLComputePipelineState> dispatch = nil;
  id<MTLComputePipelineState> histogram = nil;
  id<MTLComputePipelineState> prefix = nil;
  id<MTLComputePipelineState> base = nil;
  id<MTLComputePipelineState> scatter = nil;
  id<MTLBuffer> input_keys = nil;
  id<MTLBuffer> input_values = nil;
  id<MTLBuffer> output_keys = nil;
  id<MTLBuffer> output_values = nil;
  id<MTLBuffer> temp_keys = nil;
  id<MTLBuffer> temp_values = nil;
  id<MTLBuffer> block_counts = nil;
  id<MTLBuffer> block_offsets = nil;
  id<MTLBuffer> bucket_offsets = nil;
  id<MTLBuffer> dispatch_args = nil;
  id<MTLBuffer> status = nil;
  id<MTLBuffer> logical_count = nil;
  NSUInteger input_keys_offset = 0u;
  NSUInteger input_values_offset = 0u;
  NSUInteger output_keys_offset = 0u;
  NSUInteger output_values_offset = 0u;
  NSUInteger temp_keys_offset = 0u;
  NSUInteger temp_values_offset = 0u;
  NSUInteger block_counts_offset = 0u;
  NSUInteger block_offsets_offset = 0u;
  NSUInteger bucket_offsets_offset = 0u;
  NSUInteger logical_count_offset = 0u;
  MTLSize groups{};
  MTLSize threads{};
  bool bounded = false;
};
#endif

} // namespace rund::node::accel::detail
