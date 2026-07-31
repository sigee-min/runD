#pragma once

#include "fresh.hpp"
#include "reuse.hpp"
#include "telemetry.hpp"
#include "../../scratch.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool CreateVulkanBuffer(VulkanAdapter &adapter, const VkDeviceSize bytes,
                        const VkBufferUsageFlags usage, VulkanBuffer &buffer,
                        bool *const reused, const VulkanMemoryUse use) {
  buffer = VulkanBuffer{};
  if (bytes == 0u) {
    SetVulkanLastError(adapter, "accel_vulkan_memory_unavailable");
    return false;
  }
  if (use == VulkanMemoryUse::Scratch) {
    VulkanScratch *const scratch = ActiveVulkanScratch();
    if (scratch != nullptr) {
      return scratch->acquire(bytes, usage, buffer);
    }
  }
  const VulkanMemoryUse physical =
      use == VulkanMemoryUse::Scratch ? VulkanMemoryUse::Device : use;
  if (TakeReusableVulkanBuffer(adapter, bytes, usage, physical, buffer)) {
    if (reused != nullptr) {
      *reused = true;
    }
    RecordVulkanMemoryLease(adapter, buffer, true, physical);
    return true;
  }
  if (reused != nullptr) {
    *reused = false;
  }
  if (!CreateFreshVulkanBuffer(adapter, bytes, usage, physical, buffer)) {
    return false;
  }
  RecordVulkanMemoryLease(adapter, buffer, false, physical);
  return true;
}

#endif

} // namespace rund::node::accel::detail
