#include "../resident/access.hpp"
#include "create/telemetry.hpp"
#include "local.hpp"
#include "resident/pool.hpp"
#include "resident/storage.hpp"

#include <cstring>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
VulkanResidentOwner::~VulkanResidentOwner() {
  if (adapter == nullptr) {
    return;
  }
  VulkanResidentState &resident = VulkanResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  resident.buffers.erase(id);
  adapter = nullptr;
  id = 0u;
}

VulkanResidentStorage::~VulkanResidentStorage() {
  if (adapter == nullptr) {
    return;
  }
  VulkanResidentState &resident = VulkanResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  RetireVulkanResidentStorage(*adapter, buffer);
  adapter = nullptr;
}

void DestroyVulkanBuffer(VulkanAdapter &adapter, VulkanBuffer &buffer) {
  if (buffer.borrowed) {
    buffer = VulkanBuffer{};
    return;
  }
  ReleaseVulkanMemoryLease(adapter, buffer);
  if (buffer.memory != VK_NULL_HANDLE && buffer.mapped != nullptr) {
    vkUnmapMemory(adapter.device, buffer.memory);
  }
  if (buffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(adapter.device, buffer.buffer, nullptr);
  }
  if (buffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(adapter.device, buffer.memory, nullptr);
  }
  buffer = VulkanBuffer{};
}

bool UploadVulkanBuffer(VulkanBuffer &buffer, const void *const data,
                        const VkDeviceSize bytes) {
  if (buffer.mapped == nullptr || data == nullptr || bytes > buffer.bytes) {
    return false;
  }
  std::memcpy(buffer.mapped, data, static_cast<std::size_t>(bytes));
  return true;
}

bool ClearVulkanBuffer(VulkanBuffer &buffer, const VkDeviceSize bytes) {
  if (buffer.mapped == nullptr || bytes > buffer.bytes) {
    return false;
  }
  std::memset(buffer.mapped, 0, static_cast<std::size_t>(bytes));
  return true;
}
#endif

} // namespace rund::node::accel::detail
