#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool InjectVulkanResidencySlidingStaleDescriptorOnce(
    const std::shared_ptr<void> &prepared) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || pipeline->residency == nullptr ||
      !pipeline->residency->sliding.ready_for_submit) {
    return false;
  }
  pipeline->residency->sliding.force_stale_once.store(
      true, std::memory_order_release);
  return true;
}

bool InspectVulkanResidencySliding(
    const std::shared_ptr<void> &prepared,
    VulkanResidencySlidingDiagnostics &diagnostics) noexcept {
  diagnostics = {};
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || pipeline->residency == nullptr ||
      pipeline->adapter == nullptr) {
    return false;
  }
  const VulkanResidencySlidingGate &gate = pipeline->residency->sliding;
  diagnostics.submit_frontier_count =
      gate.submit_frontier_count.load(std::memory_order_acquire);
  diagnostics.terminal_frontier_count =
      gate.terminal_frontier_count.load(std::memory_order_acquire);
  diagnostics.gpu_result_read_count =
      gate.gpu_result_read_count.load(std::memory_order_relaxed);
  diagnostics.terminal_frontier_released =
      gate.terminal_frontier_release.load(std::memory_order_acquire);
  diagnostics.gate_quarantined =
      gate.quarantined.load(std::memory_order_acquire);
  diagnostics.adapter_quarantined =
      pipeline->adapter->residency_quarantined.load(std::memory_order_acquire);
  return true;
}

bool InjectVulkanResidencySlidingTerminalFrontierPauseOnce(
    const std::shared_ptr<void> &prepared) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || pipeline->residency == nullptr ||
      pipeline->adapter == nullptr) {
    return false;
  }
  VulkanResidencySlidingGate &gate = pipeline->residency->sliding;
  if (!gate.ready_for_submit ||
      gate.quarantined.load(std::memory_order_acquire) ||
      pipeline->adapter->residency_quarantined.load(
          std::memory_order_acquire) ||
      gate.pause_terminal_frontier_once.exchange(true,
                                                 std::memory_order_acq_rel)) {
    return false;
  }
  gate.terminal_frontier_release.store(false, std::memory_order_release);
  return true;
}

bool ReleaseVulkanResidencySlidingTerminalFrontier(
    const std::shared_ptr<void> &prepared) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || pipeline->residency == nullptr) {
    return false;
  }
  VulkanResidencySlidingGate &gate = pipeline->residency->sliding;
  if (gate.terminal_frontier_release.exchange(true,
                                              std::memory_order_acq_rel)) {
    return false;
  }
  gate.terminal_frontier_release.notify_all();
  return true;
}

#endif

} // namespace rund::node::accel::detail

namespace rund::node::accel::detail::vulkan_sliding_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void observe_submit_frontier(VulkanResidencySlidingGate &gate) noexcept {
  gate.submit_frontier_count.fetch_add(1u, std::memory_order_release);
  gate.submit_frontier_count.notify_all();
}

void pause_terminal_frontier(VulkanResidencySlidingGate &gate) noexcept {
  if (!gate.pause_terminal_frontier_once.exchange(false,
                                                  std::memory_order_acq_rel)) {
    return;
  }
  gate.terminal_frontier_count.fetch_add(1u, std::memory_order_release);
  gate.terminal_frontier_count.notify_all();
  gate.terminal_frontier_release.wait(false, std::memory_order_acquire);
}

#endif

} // namespace rund::node::accel::detail::vulkan_sliding_detail
