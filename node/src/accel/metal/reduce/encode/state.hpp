#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalReduceCommandState {
  MetalReduceEncodeResources* reduce = nullptr;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> input = nil;
  id<MTLBuffer> partial = nil;
  id<MTLBuffer> output = nil;
  id<MTLBuffer> status = nil;
  id<MTLBuffer> logical_count = nil;
  NSUInteger input_offset = 0u;
  NSUInteger partial_offset = 0u;
  NSUInteger output_offset = 0u;
  NSUInteger logical_count_offset = 0u;
};
#endif

}  // namespace rund::node::accel::detail
