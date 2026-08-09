#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalRangeState {
  MetalRangeResources *range = nullptr;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLBuffer> input = nil;
  id<MTLBuffer> output = nil;
};
#endif

} // namespace rund::node::accel::detail
