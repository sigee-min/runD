#include "../../../../include/node/accel/buffer.hpp"
#include "internal.hpp"

#include "../adapter/access.hpp"
#include "../buffer/create/telemetry.hpp"
#include "../buffer/stats.hpp"

#include <algorithm>

namespace rund::node::accel::detail::vulkan_ops_detail {

rund::RuntimeStats Stats(const rund::AccelDevice &pick) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ReadVulkanRuntimeStats(pick);
#else
  (void)pick;
  return rund::RuntimeStats{
      .outcome = {.reason = "accel_runtime_stats_unavailable"}};
#endif
}

void Reset(const rund::AccelDevice &pick) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  ResetVulkanRuntimeStats(pick);
#else
  (void)pick;
#endif
}

rund::AccelCheck
VirtualPipelineCapability(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "compute_adapter_unavailable"};
  }
  if (adapter->residency_quarantined.load(std::memory_order_acquire)) {
    return rund::AccelCheck{false, "compute_device_lost"};
  }
  // Concrete Pipeline preparation authenticates the retained selection and
  // every Pool transfer owner. This gate only publishes adapter capability.
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  return rund::AccelCheck{false, "compute_backend_unsupported"};
#endif
}

rund::node::accel::AccelMemoryStats
Memory(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return {};
  }
  std::lock_guard lock{adapter->mutex};
  const VulkanMemoryStats &memory = adapter->staging_memory;
  const std::uint64_t physical = VulkanPhysicalStaging(*adapter);
  return rund::node::accel::AccelMemoryStats{
      .staging = rund::node::accel::AccelMemoryCounter{
          .current = physical,
          .peak = std::max(physical, memory.peak),
          .cumulative = memory.cumulative,
          .reused = memory.reused}};
#else
  (void)pick;
  return {};
#endif
}

bool InjectDeviceLostOnce(const rund::AccelDevice &pick,
                          const SubmitKind kind) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  return adapter != nullptr && adapter->device_loss_fault.arm(kind);
#else
  (void)pick;
  (void)kind;
  return false;
#endif
}

bool InjectHostReadUnavailableOnce(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_host_read_once.store(true, std::memory_order_relaxed);
  return true;
#else
  (void)pick;
  return false;
#endif
}

bool InjectHostWriteUnavailableOnce(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_host_write_once.store(true, std::memory_order_relaxed);
  return true;
#else
  (void)pick;
  return false;
#endif
}

bool InjectResidencyTerminalLossOnce(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr ||
      adapter->residency_quarantined.load(std::memory_order_acquire)) {
    return false;
  }
  adapter->fault_residency_terminal_once.store(true, std::memory_order_release);
  return true;
#else
  (void)pick;
  return false;
#endif
}

bool InjectTraceUnavailableOnce(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_trace_unavailable_once.store(true, std::memory_order_relaxed);
  return true;
#else
  (void)pick;
  return false;
#endif
}

} // namespace rund::node::accel::detail::vulkan_ops_detail
