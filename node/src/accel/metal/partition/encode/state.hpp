#pragma once

#include "../local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalPartitionCommandState {
  MetalPartitionEncodeResources *partition = nullptr;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLComputePipelineState> classify = nil;
  id<MTLComputePipelineState> scatter = nil;
  id<MTLBuffer> flags = nil;
  id<MTLBuffer> values = nil;
  id<MTLBuffer> output = nil;
  id<MTLBuffer> false_bits = nil;
  id<MTLBuffer> false_offsets = nil;
  NSUInteger flags_offset = 0u;
  NSUInteger values_offset = 0u;
  NSUInteger output_offset = 0u;
  NSUInteger false_bits_offset = 0u;
  NSUInteger false_offsets_offset = 0u;
};
#endif

} // namespace rund::node::accel::detail
