#pragma once

#include <rund/counter.hpp>
#include "api.hpp"
#include "memory.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] inline const char *
VulkanBufferFailure(const VkResult result,
                    const char *const fallback) noexcept {
  switch (result) {
  case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    return "compute_device_capacity";
  case VK_ERROR_OUT_OF_HOST_MEMORY:
  case VK_ERROR_TOO_MANY_OBJECTS:
    return "compute_buffer_capacity";
  default:
    return fallback;
  }
}

[[nodiscard]] bool CreateFreshVulkanBuffer(VulkanAdapter &adapter,
                                           const VkDeviceSize bytes,
                                           const VkBufferUsageFlags usage,
                                           const VulkanMemoryUse use,
                                           VulkanBuffer &buffer) {
  const VkBufferUsageFlags effective_usage =
      use == VulkanMemoryUse::Resident
          ? usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT
          : usage;
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = bytes;
  buffer_info.usage = effective_usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  const VkResult created =
      vkCreateBuffer(adapter.device, &buffer_info, nullptr, &buffer.buffer);
  if (created != VK_SUCCESS || buffer.buffer == VK_NULL_HANDLE) {
    SetVulkanLastError(
        adapter,
        VulkanBufferFailure(created, "accel_vulkan_buffer_unavailable"));
    return false;
  }

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(adapter.device, buffer.buffer, &requirements);
  std::uint32_t memory_type = 0u;
  VkMemoryPropertyFlags memory_flags = 0u;
  if (!FindVulkanMemoryType(adapter, requirements.memoryTypeBits, use,
                            memory_type, memory_flags)) {
    DestroyVulkanBuffer(adapter, buffer);
    SetVulkanLastError(adapter, "accel_vulkan_memory_unavailable");
    return false;
  }

  VkMemoryAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = memory_type;
  const VkResult allocated =
      vkAllocateMemory(adapter.device, &allocate_info, nullptr, &buffer.memory);
  if (allocated != VK_SUCCESS || buffer.memory == VK_NULL_HANDLE) {
    DestroyVulkanBuffer(adapter, buffer);
    SetVulkanLastError(
        adapter,
        VulkanBufferFailure(allocated, "accel_vulkan_memory_unavailable"));
    return false;
  }
  const VkResult bound =
      vkBindBufferMemory(adapter.device, buffer.buffer, buffer.memory, 0u);
  if (bound != VK_SUCCESS) {
    DestroyVulkanBuffer(adapter, buffer);
    SetVulkanLastError(
        adapter, VulkanBufferFailure(bound, "accel_vulkan_memory_unavailable"));
    return false;
  }
  const VkResult mapped = use == VulkanMemoryUse::Staging
                              ? vkMapMemory(adapter.device, buffer.memory, 0u,
                                            bytes, 0u, &buffer.mapped)
                              : VK_SUCCESS;
  if (mapped != VK_SUCCESS ||
      (use == VulkanMemoryUse::Staging && buffer.mapped == nullptr)) {
    DestroyVulkanBuffer(adapter, buffer);
    SetVulkanLastError(adapter, VulkanBufferFailure(
                                    mapped, "accel_vulkan_memory_unavailable"));
    return false;
  }

  buffer.bytes = bytes;
  buffer.usage = effective_usage;
  buffer.memory_flags = memory_flags;
  buffer.memory_use = use;
  ::rund::detail::counter::Accumulate(adapter.buffer_allocation_count, 1u);
  return true;
}

#endif

} // namespace rund::node::accel::detail
