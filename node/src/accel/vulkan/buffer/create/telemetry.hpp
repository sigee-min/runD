#pragma once

#include "../../../kernel/memory.hpp"
#include "../local.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline std::uint64_t
VulkanPhysicalStaging(const VulkanAdapter &adapter) noexcept {
  return ::rund::detail::counter::SaturatingAdd(
      adapter.staging_memory.current, adapter.staging_memory.pooled);
}

inline void RecordVulkanPhysicalStaging(VulkanAdapter &adapter) noexcept {
  VulkanMemoryStats &stats = adapter.staging_memory;
  stats.peak = std::max(stats.peak, VulkanPhysicalStaging(adapter));
}

[[nodiscard]] inline PreparedMemory VulkanPreparedMemory(
    const VulkanMemoryStats before, const VulkanMemoryStats after,
    const std::uint64_t budget) noexcept {
  return PreparedMemory{
      .current = ::rund::detail::counter::Delta(before.current, after.current),
      .peak = ::rund::detail::counter::Delta(before.current, after.current),
      .cumulative =
          ::rund::detail::counter::Delta(before.cumulative, after.cumulative),
      .reused = ::rund::detail::counter::Delta(before.reused, after.reused),
      .budget = budget};
}

inline void RecordVulkanMemoryLease(VulkanAdapter &adapter,
                                    VulkanBuffer &buffer,
                                    const bool reused,
                                    const VulkanMemoryUse use) noexcept {
  buffer.memory_use = use;
  buffer.memory_lease = use == VulkanMemoryUse::Staging;
  if (!buffer.memory_lease) {
    return;
  }
  VulkanMemoryStats &stats = adapter.staging_memory;
  ::rund::detail::counter::Accumulate(stats.current, buffer.bytes);
  ::rund::detail::counter::Accumulate(stats.cumulative, buffer.bytes);
  if (reused) {
    ::rund::detail::counter::Accumulate(stats.reused, buffer.bytes);
  }
  RecordVulkanPhysicalStaging(adapter);
}

inline void ReleaseVulkanMemoryLease(VulkanAdapter &adapter,
                                     VulkanBuffer &buffer) noexcept {
  if (!buffer.memory_lease) {
    return;
  }
  VulkanMemoryStats &stats = adapter.staging_memory;
  ::rund::detail::counter::Release(stats.current, buffer.bytes);
  buffer.memory_lease = false;
}
#endif

} // namespace rund::node::accel::detail
