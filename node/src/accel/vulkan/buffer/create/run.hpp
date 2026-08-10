#pragma once

#include "../../scratch.hpp"
#include "fresh.hpp"
#include "reuse.hpp"
#include "telemetry.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool CreateVulkanBuffer(VulkanAdapter &adapter, const VkDeviceSize bytes,
                        const VkBufferUsageFlags usage, VulkanBuffer &buffer,
                        bool *const reused, const VulkanMemoryUse use,
                        const std::uint64_t exact_storage_bytes) {
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
  if (TakeReusableVulkanBuffer(adapter, bytes, usage, physical, buffer,
                               exact_storage_bytes)) {
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
  if (exact_storage_bytes != 0u &&
      buffer.allocated_bytes != exact_storage_bytes) {
    DestroyVulkanBuffer(adapter, buffer);
    SetVulkanLastError(adapter, "accel_vulkan_memory_unavailable");
    return false;
  }
  RecordVulkanMemoryLease(adapter, buffer, false, physical);
  return true;
}

#endif

} // namespace rund::node::accel::detail
