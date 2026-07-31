#pragma once

#include "input.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline bool
PrepareVulkanOutputBuffer(VulkanAdapter &adapter,
                          const rund::kernel::ComputePlan &plan,
                          const rund::kernel::ComputeDispatchWindow &window,
                          VulkanWindowBuffers &out) {
  if (!rund::kernel::checked::mul(window.tile_count,
                                  plan.output_bytes_per_tile)) {
    SetVulkanLastError(adapter, "compute_dispatch_overflow");
    return false;
  }
  const rund::kernel::u64 output_byte_count =
      window.tile_count * plan.output_bytes_per_tile;
  if (!ToSize(output_byte_count, out.output_size)) {
    SetVulkanLastError(adapter, "compute_dispatch_overflow");
    return false;
  }
  if (out.resident) {
    return true;
  }
  VulkanBuffer output_raw{};
  if (!CreateVulkanBuffer(adapter, static_cast<VkDeviceSize>(output_byte_count),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, output_raw)) {
    return false;
  }
  out.output = ScopedBuffer{adapter, output_raw,
                            static_cast<VkDeviceSize>(out.output_size)};
  return true;
}
#endif

} // namespace rund::node::accel::detail
