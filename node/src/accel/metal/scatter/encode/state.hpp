#pragma once

#include "../local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalScatterCommandState {
  MetalScatterEncodeResources* scatter = nullptr;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> values = nil;
  id<MTLBuffer> indices = nil;
  id<MTLBuffer> output = nil;
  id<MTLBuffer> status = nil;
};
#endif

}  // namespace rund::node::accel::detail
