#pragma once

#include "../../adapter/api.hpp"
#include "../../resident/access.hpp"
#include "../local.hpp"

#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// VulkanResidentState::mutex owns every call in this file. The fixed array
// makes final public-handle release allocation-free even when it occurs during
// prepared resource destruction.
[[nodiscard]] inline bool
TakeVulkanResidentStorage(VulkanAdapter &adapter, const VkDeviceSize bytes,
                          const VkBufferUsageFlags usage,
                          VulkanBuffer &buffer) noexcept {
  const VkBufferUsageFlags effective_usage = usage |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VulkanResidentState &resident = VulkanResidents(adapter);
  std::size_t best = kVulkanPoolCapacity;
  for (std::size_t index = 0u; index < resident.pool_size; ++index) {
    const VulkanBuffer &candidate = resident.pool[index];
    if (candidate.usage != effective_usage || candidate.bytes < bytes) {
      continue;
    }
    if (best == kVulkanPoolCapacity ||
        candidate.bytes < resident.pool[best].bytes) {
      best = index;
    }
  }
  if (best == kVulkanPoolCapacity) {
    return false;
  }
  buffer = resident.pool[best];
  ::rund::detail::counter::Release(resident.pool_bytes, buffer.bytes);
  --resident.pool_size;
  for (std::size_t index = best; index < resident.pool_size; ++index) {
    resident.pool[index] = resident.pool[index + 1u];
  }
  resident.pool[resident.pool_size] = VulkanBuffer{};
  return true;
}

inline void EvictVulkanResidentStorage(VulkanAdapter &adapter) noexcept {
  VulkanResidentState &resident = VulkanResidents(adapter);
  if (resident.pool_size == 0u) {
    return;
  }
  VulkanBuffer evicted = resident.pool[0u];
  ::rund::detail::counter::Release(resident.pool_bytes, evicted.bytes);
  --resident.pool_size;
  for (std::size_t index = 0u; index < resident.pool_size; ++index) {
    resident.pool[index] = resident.pool[index + 1u];
  }
  resident.pool[resident.pool_size] = VulkanBuffer{};
  DestroyVulkanBuffer(adapter, evicted);
}

inline void RetireVulkanResidentStorage(VulkanAdapter &adapter,
                                        VulkanBuffer &buffer) noexcept {
  VulkanResidentState &resident = VulkanResidents(adapter);
  if (buffer.buffer == VK_NULL_HANDLE || buffer.memory == VK_NULL_HANDLE ||
      buffer.memory_use != VulkanMemoryUse::Resident) {
    DestroyVulkanBuffer(adapter, buffer);
    return;
  }
  const std::uint64_t limit = VulkanPoolLimit(adapter.caps.staging_bytes);
  if (buffer.bytes > limit) {
    DestroyVulkanBuffer(adapter, buffer);
    return;
  }
  while (resident.pool_size != 0u &&
         (resident.pool_size >= kVulkanPoolCapacity ||
          !FitsVulkanPool(resident.pool_bytes, buffer.bytes, limit))) {
    EvictVulkanResidentStorage(adapter);
  }
  if (!FitsVulkanPool(resident.pool_bytes, buffer.bytes, limit)) {
    DestroyVulkanBuffer(adapter, buffer);
    return;
  }
  resident.pool[resident.pool_size] = buffer;
  ++resident.pool_size;
  ::rund::detail::counter::Accumulate(resident.pool_bytes, buffer.bytes);
  buffer = VulkanBuffer{};
}

inline void DestroyVulkanResidentStorage(VulkanAdapter &adapter) noexcept {
  VulkanResidentState &resident = VulkanResidents(adapter);
  for (std::size_t index = 0u; index < resident.pool_size; ++index) {
    DestroyVulkanBuffer(adapter, resident.pool[index]);
  }
  resident.pool_size = 0u;
  resident.pool_bytes = 0u;
}

#endif

} // namespace rund::node::accel::detail
