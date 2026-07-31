#include <accel/check.hpp>

#include "local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] id<MTLBuffer> AsMTLBuffer(const std::shared_ptr<void> &buffer) {
  return (__bridge id<MTLBuffer>)buffer.get();
}

[[nodiscard]] id<MTLComputePipelineState>
AsMTLPipeline(const std::shared_ptr<void> &pipeline) {
  return (__bridge id<MTLComputePipelineState>)pipeline.get();
}

void DispatchGroups(id<MTLComputeCommandEncoder> encoder,
                    const rund::kernel::u64 groups) {
  [encoder
       dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(groups), 1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(kSegmentedScanWidth, 1u, 1u)];
}

} // namespace
#endif

rund::AccelCheck
EncodeMetalSegmentedScan(MetalAdapter &adapter,
                         const std::shared_ptr<void> &resources,
                         void *const command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const scan =
      static_cast<MetalSegmentedScanEncodeResources *>(resources.get());
  id<MTLComputeCommandEncoder> encoder =
      (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  if (scan == nullptr || scan->adapter != &adapter || encoder == nil ||
      scan->input.device_buffer == nullptr ||
      scan->heads.device_buffer == nullptr ||
      scan->output.device_buffer == nullptr ||
      scan->offsets.buffer == nullptr || scan->first_heads.buffer == nullptr ||
      scan->status.buffer == nullptr || scan->block == nullptr ||
      scan->prefix == nullptr || scan->offset == nullptr ||
      (scan->plan.pass_count != 1u && scan->plan.pass_count != 2u)) {
    SetMetalLastError(adapter, "compute_segmented_scan_invalid");
    return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
  }
  if ([AsMTLPipeline(scan->block) maxTotalThreadsPerThreadgroup] <
          kSegmentedScanWidth ||
      [AsMTLPipeline(scan->prefix) maxTotalThreadsPerThreadgroup] <
          kSegmentedScanWidth ||
      [AsMTLPipeline(scan->offset) maxTotalThreadsPerThreadgroup] <
          kSegmentedScanWidth) {
    SetMetalLastError(adapter, "compute_segmented_scan_invalid");
    return rund::AccelCheck{false, "compute_segmented_scan_invalid"};
  }
  const SegmentedScanParams params{
      scan->plan.element_count, scan->plan.block_size, scan->plan.block_count,
      scan->plan.op == rund::kernel::SegmentedScanOp::InclusiveSum ? 1u : 0u,
      0u};
  [encoder setComputePipelineState:AsMTLPipeline(scan->block)];
  [encoder setBuffer:AsMTLBuffer(scan->input.device_buffer)
              offset:static_cast<NSUInteger>(scan->input.ref.offset_bytes)
             atIndex:0u];
  [encoder setBuffer:AsMTLBuffer(scan->heads.device_buffer)
              offset:static_cast<NSUInteger>(scan->heads.ref.offset_bytes)
             atIndex:1u];
  [encoder setBuffer:AsMTLBuffer(scan->output.device_buffer)
              offset:static_cast<NSUInteger>(scan->output.ref.offset_bytes)
             atIndex:2u];
  [encoder setBuffer:AsMTLBuffer(scan->offsets.buffer)
               offset:static_cast<NSUInteger>(scan->offsets.offset)
              atIndex:3u];
  [encoder setBuffer:AsMTLBuffer(scan->first_heads.buffer)
               offset:static_cast<NSUInteger>(scan->first_heads.offset)
              atIndex:4u];
  [encoder setBuffer:AsMTLBuffer(scan->status.buffer) offset:0u atIndex:5u];
  [encoder setBytes:&params length:sizeof(params) atIndex:6u];
  DispatchGroups(encoder, scan->plan.block_count);
  if (scan->plan.pass_count == 1u) {
    return rund::AccelCheck{true, "ok"};
  }
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

  [encoder setComputePipelineState:AsMTLPipeline(scan->prefix)];
  [encoder setBuffer:AsMTLBuffer(scan->offsets.buffer)
               offset:static_cast<NSUInteger>(scan->offsets.offset)
              atIndex:0u];
  [encoder setBuffer:AsMTLBuffer(scan->first_heads.buffer)
               offset:static_cast<NSUInteger>(scan->first_heads.offset)
              atIndex:1u];
  [encoder setBuffer:AsMTLBuffer(scan->status.buffer) offset:0u atIndex:2u];
  [encoder setBytes:&params length:sizeof(params) atIndex:3u];
  DispatchGroups(encoder, 1u);
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

  [encoder setComputePipelineState:AsMTLPipeline(scan->offset)];
  [encoder setBuffer:AsMTLBuffer(scan->input.device_buffer)
              offset:static_cast<NSUInteger>(scan->input.ref.offset_bytes)
             atIndex:0u];
  [encoder setBuffer:AsMTLBuffer(scan->output.device_buffer)
              offset:static_cast<NSUInteger>(scan->output.ref.offset_bytes)
             atIndex:1u];
  [encoder setBuffer:AsMTLBuffer(scan->offsets.buffer)
               offset:static_cast<NSUInteger>(scan->offsets.offset)
              atIndex:2u];
  [encoder setBuffer:AsMTLBuffer(scan->first_heads.buffer)
               offset:static_cast<NSUInteger>(scan->first_heads.offset)
              atIndex:3u];
  [encoder setBuffer:AsMTLBuffer(scan->status.buffer) offset:0u atIndex:4u];
  [encoder setBytes:&params length:sizeof(params) atIndex:5u];
  DispatchGroups(encoder, scan->plan.block_count);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
