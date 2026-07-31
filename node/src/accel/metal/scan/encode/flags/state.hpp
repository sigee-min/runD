#pragma once

#include "../state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalFlagScanEncodeState {
  std::shared_ptr<void> block_handle{};
  std::shared_ptr<void> prefix_handle{};
  std::shared_ptr<void> offset_handle{};
  id<MTLComputePipelineState> block = nil;
  id<MTLComputePipelineState> prefix = nil;
  id<MTLComputePipelineState> offset = nil;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLBuffer> flags = nil;
  id<MTLBuffer> output = nil;
  id<MTLBuffer> totals = nil;
  id<MTLBuffer> status = nil;
  NSUInteger flags_offset = 0u;
  NSUInteger output_offset = 0u;
  NSUInteger totals_offset = 0u;
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 block_size = 0u;
  rund::kernel::u64 block_count = 0u;
  NSUInteger block_threads = 0u;
  NSUInteger prefix_threads = 0u;
};
#endif

} // namespace rund::node::accel::detail
