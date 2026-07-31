#include <accel/check.hpp>

#include "local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

rund::AccelCheck EncodeMetalHistogram(MetalAdapter &adapter,
                                      const std::shared_ptr<void> &resources,
                                      void *const command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const histogram =
      static_cast<MetalHistogramEncodeResources *>(resources.get());
  id<MTLComputeCommandEncoder> encoder =
      (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  if (histogram == nullptr || histogram->adapter != &adapter ||
      encoder == nil) {
    SetMetalLastError(adapter, "compute_histogram_invalid");
    return rund::AccelCheck{false, "compute_histogram_invalid"};
  }
  id<MTLComputePipelineState> clear_pipeline =
      (__bridge id<MTLComputePipelineState>)histogram->pipelines.clear.get();
  id<MTLComputePipelineState> count_pipeline =
      (__bridge id<MTLComputePipelineState>)histogram->pipelines.count.get();
  id<MTLBuffer> bins =
      (__bridge id<MTLBuffer>)histogram->bins.device_buffer.get();
  id<MTLBuffer> counts =
      (__bridge id<MTLBuffer>)histogram->counts.device_buffer.get();
  id<MTLBuffer> status = (__bridge id<MTLBuffer>)histogram->status.buffer.get();
  if (clear_pipeline == nil || count_pipeline == nil || bins == nil ||
      counts == nil || status == nil) {
    SetMetalLastError(adapter, "compute_histogram_invalid");
    return rund::AccelCheck{false, "compute_histogram_invalid"};
  }
  const HistogramParams params{histogram->plan.element_count,
                               histogram->plan.bin_count};
  [encoder setComputePipelineState:clear_pipeline];
  [encoder setBuffer:counts
              offset:static_cast<NSUInteger>(histogram->counts.ref.offset_bytes)
             atIndex:0u];
  [encoder setBuffer:status offset:0u atIndex:1u];
  [encoder setBytes:&params length:sizeof(params) atIndex:2u];
  [encoder dispatchThreads:MTLSizeMake(
                               static_cast<NSUInteger>(params.bin_count), 1u,
                               1u)
      threadsPerThreadgroup:MTLSizeMake(kHistogramThreadgroupSize, 1u, 1u)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [encoder setComputePipelineState:count_pipeline];
  [encoder setBuffer:bins
              offset:static_cast<NSUInteger>(histogram->bins.ref.offset_bytes)
             atIndex:0u];
  [encoder setBuffer:counts
              offset:static_cast<NSUInteger>(histogram->counts.ref.offset_bytes)
             atIndex:1u];
  [encoder setBuffer:status offset:0u atIndex:2u];
  [encoder setBytes:&params length:sizeof(params) atIndex:3u];
  [encoder dispatchThreads:MTLSizeMake(
                               static_cast<NSUInteger>(params.element_count),
                               1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(kHistogramThreadgroupSize, 1u, 1u)];
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
