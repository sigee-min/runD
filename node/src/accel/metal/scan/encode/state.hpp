#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

#include <memory>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalScanEncodeState {
  std::shared_ptr<void> block_handle{};
  std::shared_ptr<void> prefix_handle{};
  std::shared_ptr<void> offset_handle{};
  id<MTLComputePipelineState> block = nil;
  id<MTLComputePipelineState> prefix = nil;
  id<MTLComputePipelineState> offset = nil;
  id<MTLComputeCommandEncoder> encoder = nil;
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u64 block_size = 0u;
  rund::kernel::u64 block_count = 0u;
  NSUInteger block_threads = 0u;
  NSUInteger prefix_threads = 0u;
};

[[nodiscard]] rund::AccelCheck PrepareMetalScanEncodeState(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *input_buffer, void *output_buffer,
    void *totals_buffer, void *status_buffer, void *command_encoder,
    MetalScanEncodeState &state, const std::shared_ptr<void> *block = nullptr,
    const std::shared_ptr<void> *prefix = nullptr,
    const std::shared_ptr<void> *offset = nullptr);
#endif

} // namespace rund::node::accel::detail
