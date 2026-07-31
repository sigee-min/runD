#pragma once

#include "../../../descriptor.hpp"

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool CollectiveDescriptorSlotOk(
    VulkanAdapter& adapter,
    const VulkanCollectivePipeline& pipeline,
    const std::uint32_t descriptor_count,
    const std::uint64_t slot) {
  if (pipeline.descriptor_count == descriptor_count &&
      pipeline.descriptor_set_layout != VK_NULL_HANDLE &&
      slot < static_cast<std::uint64_t>(
                 std::numeric_limits<std::size_t>::max()) &&
      slot < static_cast<std::uint64_t>(
                 std::numeric_limits<std::uint32_t>::max())) {
    return true;
  }
  SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
  return false;
}

#endif

}  // namespace rund::node::accel::detail
