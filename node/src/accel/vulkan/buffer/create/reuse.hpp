#pragma once

#include "../local.hpp"
#include "../pool.hpp"
#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool TakeReusableVulkanBuffer(
    VulkanAdapter &adapter, const VkDeviceSize bytes,
    const VkBufferUsageFlags usage, const VulkanMemoryUse use,
    VulkanBuffer &buffer) {
  const VkBufferUsageFlags effective_usage =
      use == VulkanMemoryUse::Resident
          ? usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT
          : usage;
  auto best = adapter.reusable_buffers.end();
  for (auto it = adapter.reusable_buffers.begin();
       it != adapter.reusable_buffers.end(); ++it) {
    if (it->usage != effective_usage || it->memory_use != use ||
        it->bytes < bytes) {
      continue;
    }
    if (best == adapter.reusable_buffers.end() || it->bytes < best->bytes) {
      best = it;
    }
  }
  if (best == adapter.reusable_buffers.end()) {
    return false;
  }
  buffer = *best;
  ::rund::detail::counter::Release(adapter.reusable_buffer_bytes,
                                   buffer.bytes);
  if (buffer.memory_use == VulkanMemoryUse::Staging) {
    ::rund::detail::counter::Release(adapter.staging_memory.pooled,
                                     buffer.bytes);
  }
  adapter.reusable_buffers.erase(best);
  ::rund::detail::counter::Accumulate(adapter.buffer_reuse_hit_count, 1u);
  return true;
}

#endif

} // namespace rund::node::accel::detail
