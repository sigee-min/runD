#include "internal.hpp"

#include <algorithm>

namespace rund::node::accel::detail::metal_device_vsm::prepare {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

bool RecordGraph(Owner &owner, MetalAdapter &adapter) noexcept {
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)owner.pipeline.get();
  if (device == nil || pipeline == nil || owner.params == nullptr ||
      owner.graph.table == nullptr || owner.result == nullptr ||
      owner.graph.owner_count == 0u || owner.graph.endpoint_count == 0u) {
    return false;
  }
  id<MTLIndirectCommandBuffer> icb = AllocateMetalPipelineIcb(device, 1u);
  id<MTLIndirectComputeCommand> command =
      icb == nil ? nil : [icb indirectComputeCommandAtIndex:0u];
  if (icb == nil || command == nil) {
    return false;
  }
  [command setComputePipelineState:pipeline];
  [command setKernelBuffer:(__bridge id<MTLBuffer>)owner.params.get()
                    offset:0u
                   atIndex:0u];
  [command setKernelBuffer:(__bridge id<MTLBuffer>)owner.graph.table.get()
                    offset:0u
                   atIndex:1u];
  [command setKernelBuffer:(__bridge id<MTLBuffer>)owner.result.get()
                    offset:0u
                   atIndex:2u];
  std::size_t binding = 3u;
  for (std::size_t slot = 0u; slot < owner.graph.owner_count; ++slot) {
    for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
         ++bank) {
      const auto &row = owner.graph.owners[slot][bank];
      if (!row.check.ok || row.device_buffer == nullptr) {
        return false;
      }
      [command setKernelBuffer:(__bridge id<MTLBuffer>)row.device_buffer.get()
                        offset:static_cast<NSUInteger>(row.ref.offset_bytes)
                       atIndex:static_cast<NSUInteger>(binding++)];
    }
  }
  for (std::size_t endpoint = 0u; endpoint < owner.graph.endpoint_count;
       ++endpoint) {
    const auto &row = owner.graph.endpoints[endpoint];
    if (!row.check.ok || row.device_buffer == nullptr) {
      return false;
    }
    [command setKernelBuffer:(__bridge id<MTLBuffer>)row.device_buffer.get()
                      offset:static_cast<NSUInteger>(row.ref.offset_bytes)
                     atIndex:static_cast<NSUInteger>(binding++)];
  }
  const NSUInteger threads = std::max<NSUInteger>(
      1u, std::min<NSUInteger>(256u, [pipeline maxTotalThreadsPerThreadgroup]));
  [command concurrentDispatchThreads:MTLSizeMake(threads, 1u, 1u)
               threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
  owner.graph_icb = RetainMetalObject((__bridge void *)icb);
  owner.graph_pipeline = owner.pipeline;
  owner.graph_digest = owner.proof->graph_resident.digest;
  return owner.graph_icb != nullptr;
}

#endif

} // namespace rund::node::accel::detail::metal_device_vsm::prepare
