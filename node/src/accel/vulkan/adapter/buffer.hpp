#pragma once

#include <cstdint>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanAdapter;

enum class VulkanMemoryUse : std::uint8_t {
  Staging,
  Resident,
  Device,
  Scratch,
};

struct VulkanMemoryStats final {
  std::uint64_t current{};
  std::uint64_t pooled{};
  std::uint64_t peak{};
  std::uint64_t cumulative{};
  std::uint64_t reused{};
};

struct VulkanBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize bytes = 0u;
  VkBufferUsageFlags usage = 0u;
  VkMemoryPropertyFlags memory_flags = 0u;
  void* mapped = nullptr;
  VkDeviceSize offset = 0u;
  VulkanMemoryUse memory_use{VulkanMemoryUse::Staging};
  bool memory_lease{};
  bool borrowed{};
};

#endif  // defined(RUND_NODE_HAVE_VULKAN_SDK)

}  // namespace rund::node::accel::detail
