#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalGatherCommandState {
  MetalGatherEncodeResources* gather = nullptr;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLComputePipelineState> control_pipeline = nil;
  id<MTLComputePipelineState> gather_pipeline = nil;
  id<MTLBuffer> values = nil;
  id<MTLBuffer> indices = nil;
  id<MTLBuffer> logical_count = nil;
  id<MTLBuffer> output = nil;
  id<MTLBuffer> status = nil;
  id<MTLBuffer> indirect = nil;
};
#endif

}  // namespace rund::node::accel::detail
