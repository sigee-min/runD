#include "pool.hpp"
#include "create/telemetry.hpp"
#include "local.hpp"

#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void RemoveVulkanPoolFront(VulkanAdapter &adapter) {
  VulkanBuffer &front = adapter.reusable_buffers.front();
  ::rund::detail::counter::Release(adapter.reusable_buffer_bytes,
                                   front.allocated_bytes);
  if (front.memory_use == VulkanMemoryUse::Staging) {
    ::rund::detail::counter::Release(adapter.staging_memory.pooled,
                                     front.allocated_bytes);
  }
  DestroyVulkanBuffer(adapter, front);
  adapter.reusable_buffers.erase(adapter.reusable_buffers.begin());
}

} // namespace

void ReleaseVulkanBuffer(VulkanAdapter &adapter, VulkanBuffer &buffer) {
  if (buffer.borrowed) {
    buffer = VulkanBuffer{};
    return;
  }
  ReleaseVulkanMemoryLease(adapter, buffer);
  if (buffer.buffer == VK_NULL_HANDLE || buffer.memory == VK_NULL_HANDLE ||
      (buffer.memory_use == VulkanMemoryUse::Staging &&
       buffer.mapped == nullptr)) {
    DestroyVulkanBuffer(adapter, buffer);
    return;
  }
  const std::uint64_t limit = VulkanPoolLimit(adapter.caps.staging_bytes);
  if (buffer.allocated_bytes > limit) {
    DestroyVulkanBuffer(adapter, buffer);
    return;
  }
  while (!adapter.reusable_buffers.empty() &&
         (adapter.reusable_buffers.size() >= kVulkanPoolCapacity ||
          !FitsVulkanPool(adapter.reusable_buffer_bytes, buffer.allocated_bytes,
                          limit))) {
    RemoveVulkanPoolFront(adapter);
  }
  if (!FitsVulkanPool(adapter.reusable_buffer_bytes, buffer.allocated_bytes,
                      limit)) {
    DestroyVulkanBuffer(adapter, buffer);
    return;
  }
  ::rund::detail::counter::Accumulate(adapter.reusable_buffer_bytes,
                                      buffer.allocated_bytes);
  if (buffer.memory_use == VulkanMemoryUse::Staging) {
    ::rund::detail::counter::Accumulate(adapter.staging_memory.pooled,
                                        buffer.allocated_bytes);
  }
  adapter.reusable_buffers.push_back(buffer);
  RecordVulkanPhysicalStaging(adapter);
  buffer = VulkanBuffer{};
}
#endif

} // namespace rund::node::accel::detail
