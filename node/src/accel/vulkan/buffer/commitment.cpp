#include "local.hpp"

#include "transfer/range.hpp"

#include <cstdint>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] std::uint64_t
VulkanStorageRequirement(const rund::AccelDevice &pick,
                         const std::uint64_t logical_bytes,
                         const VkBufferUsageFlags usage) noexcept {
  if (!VulkanPickOwnsAdapter(pick)) {
    return 0u;
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  VkDeviceSize bytes = 0u;
  if (adapter == nullptr || !AlignVulkanTransferStorage(logical_bytes, bytes)) {
    return 0u;
  }
  std::lock_guard lock{adapter->mutex};
  const VkBufferCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  VkBuffer buffer = VK_NULL_HANDLE;
  if (vkCreateBuffer(adapter->device, &info, nullptr, &buffer) != VK_SUCCESS ||
      buffer == VK_NULL_HANDLE) {
    return 0u;
  }
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(adapter->device, buffer, &requirements);
  vkDestroyBuffer(adapter->device, buffer, nullptr);
  return requirements.size >= bytes ? requirements.size : 0u;
}

} // namespace
#endif

std::uint64_t
VulkanBufferStorageBytes(const rund::AccelDevice &pick,
                         const std::uint64_t logical_bytes) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return VulkanStorageRequirement(pick, logical_bytes,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
#else
  (void)pick;
  (void)logical_bytes;
  return 0u;
#endif
}

std::uint64_t
VulkanPipelineTransferStorageBytes(const rund::AccelDevice &pick,
                                   const std::uint64_t logical_bytes) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return VulkanStorageRequirement(pick, logical_bytes,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
#else
  (void)pick;
  (void)logical_bytes;
  return 0u;
#endif
}

} // namespace rund::node::accel::detail
