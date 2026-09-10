#include "../model.hpp"

#include "../../../../../src/accel/kernel/memory.hpp"
#include "../../../../../src/accel/metal/buffer/owner.hpp"
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "../../../../../src/accel/vulkan/buffer/create/telemetry.hpp"
#include "../../../../../src/accel/vulkan/buffer/pool.hpp"
#include "../../../../../src/accel/vulkan/buffer/resident/pool.hpp"
#include "../../../../../src/accel/vulkan/resident/state.hpp"
#endif

namespace rund_node_memory_contract {

int CheckMetalMemoryModel() {
  using namespace rund::node::accel::detail;
  const PreparedMemory fresh = MetalBufferMemory(
      MetalRuntimeBuffer{.bytes = 32u, .allocated_bytes = 64u}, 1024u);
  const PreparedMemory reused = MetalBufferMemory(
      MetalRuntimeBuffer{.bytes = 32u, .allocated_bytes = 96u, .reused = true},
      1024u);
  return fresh.current == 64u && fresh.peak == 64u && fresh.cumulative == 64u &&
                 fresh.reused == 0u && fresh.budget == 1024u &&
                 reused.current == 96u && reused.peak == 96u &&
                 reused.cumulative == 96u && reused.reused == 96u &&
                 reused.budget == 1024u
             ? 0
             : 1;
}

int CheckVulkanMemoryModel() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  const PreparedMemory prepared = VulkanPreparedMemory(
      VulkanMemoryStats{.current = 32u, .cumulative = 64u, .reused = 8u},
      VulkanMemoryStats{.current = 96u, .cumulative = 192u, .reused = 72u},
      1024u);
  if (prepared.current != 64u || prepared.peak != 64u ||
      prepared.cumulative != 128u || prepared.reused != 64u ||
      prepared.budget != 1024u) {
    return 1;
  }
  constexpr std::uint64_t staging_budget = 1024u * 1024u;
  constexpr std::uint64_t pool_limit = staging_budget * kVulkanPoolCapacity;
  if (VulkanPoolLimit(staging_budget) != pool_limit ||
      !FitsVulkanPool(pool_limit - 64u, 64u, pool_limit) ||
      FitsVulkanPool(pool_limit - 63u, 64u, pool_limit) ||
      FitsVulkanPool(0u, pool_limit + 1u, pool_limit)) {
    return 6;
  }
  VulkanAdapter physical{};
  physical.staging_memory.pooled = 96u;
  VulkanBuffer physical_buffer{.bytes = 32u, .allocated_bytes = 64u};
  RecordVulkanMemoryLease(physical, physical_buffer, false,
                          VulkanMemoryUse::Staging);
  if (VulkanPhysicalStaging(physical) != 160u ||
      physical.staging_memory.peak != 160u) {
    return 7;
  }
  ReleaseVulkanMemoryLease(physical, physical_buffer);
  if (VulkanPhysicalStaging(physical) != 96u ||
      physical.staging_memory.peak != 160u) {
    return 8;
  }
  VulkanAdapter adapter{};
  VulkanBuffer buffer{.bytes = 64u, .allocated_bytes = 96u};
  RecordVulkanMemoryLease(adapter, buffer, false, VulkanMemoryUse::Staging);
  if (adapter.staging_memory.current != 96u ||
      adapter.staging_memory.peak != 96u ||
      adapter.staging_memory.cumulative != 96u ||
      adapter.staging_memory.reused != 0u) {
    return 2;
  }
  ReleaseVulkanMemoryLease(adapter, buffer);
  RecordVulkanMemoryLease(adapter, buffer, true, VulkanMemoryUse::Staging);
  ReleaseVulkanMemoryLease(adapter, buffer);
  if (adapter.staging_memory.current != 0u ||
      adapter.staging_memory.cumulative != 192u ||
      adapter.staging_memory.reused != 96u) {
    return 3;
  }
  RecordVulkanMemoryLease(adapter, buffer, true, VulkanMemoryUse::Resident);
  if (buffer.memory_lease || adapter.staging_memory.cumulative != 192u) {
    return 4;
  }
  adapter.staging_memory = VulkanMemoryStats{.current = kCounterMaximum - 1u,
                                             .peak = kCounterMaximum - 1u,
                                             .cumulative = kCounterMaximum - 1u,
                                             .reused = kCounterMaximum - 1u};
  buffer.bytes = 1u;
  buffer.allocated_bytes = 2u;
  RecordVulkanMemoryLease(adapter, buffer, true, VulkanMemoryUse::Staging);
  ReleaseVulkanMemoryLease(adapter, buffer);
  if (adapter.staging_memory.current != kCounterMaximum ||
      adapter.staging_memory.peak != kCounterMaximum ||
      adapter.staging_memory.cumulative != kCounterMaximum ||
      adapter.staging_memory.reused != kCounterMaximum) {
    return 5;
  }
  VulkanAdapter resident_adapter{};
  resident_adapter.resident = std::make_unique<VulkanResidentState>();
  VulkanResidentState &resident = *resident_adapter.resident;
  resident.pool[0u] = VulkanBuffer{
      .buffer = reinterpret_cast<VkBuffer>(1u),
      .memory = reinterpret_cast<VkDeviceMemory>(1u),
      .bytes = 128u,
      .capacity_bytes = 128u,
      .allocated_bytes = 256u,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      .memory_use = VulkanMemoryUse::Resident,
  };
  resident.pool[1u] = resident.pool[0u];
  resident.pool[1u].allocated_bytes = 192u;
  resident.pool_size = 2u;
  resident.pool_bytes = 448u;
  VulkanBuffer exact{};
  if (!TakeVulkanResidentStorage(resident_adapter, 64u,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 VulkanMemoryUse::Resident, exact, 192u) ||
      exact.allocated_bytes != 192u || resident.pool_size != 1u ||
      resident.pool[0u].allocated_bytes != 256u ||
      resident.pool_bytes != 256u) {
    return 9;
  }
  exact = {};
  resident.pool[0u] = {};
  resident.pool_size = 0u;
  resident.pool_bytes = 0u;
#endif
  return 0;
}

} // namespace rund_node_memory_contract
