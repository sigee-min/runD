#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalStencilCommandState {
  MetalStencilEncodeResources *stencil = nullptr;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> input = nil;
  id<MTLBuffer> output = nil;
  std::uint32_t workgroups = 0u;
};
#endif

} // namespace rund::node::accel::detail
