#include "../../../../adapter/error.hpp"

#include "internal.hpp"

#include <algorithm>
#include <atomic>
#include <limits>

namespace rund::node::accel::detail::vulkan_sliding_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool queue_gate_submission(VulkanAdapter &adapter,
                           VulkanResidencySlidingGate &gate,
                           const std::span<const VkCommandBuffer> commands,
                           const VkFence fence,
                           const KernelCompletion completion, void *const user,
                           const bool collect_timing) noexcept {
  if (commands.empty() || gate.ready == VK_NULL_HANDLE ||
      gate.next_ready_value == 0u ||
      gate.next_ready_value == std::numeric_limits<std::uint64_t>::max() ||
      adapter.timeline.signal == nullptr || completion == nullptr ||
      user == nullptr || fence == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_selection_unavailable");
    return false;
  }
  std::unique_lock completion_lock{adapter.completion_mutex};
  if (adapter.completion_stop ||
      adapter.pending_size == adapter.pending.size()) {
    SetVulkanLastError(adapter, "accel_vulkan_command_capacity");
    return false;
  }
  const VkResult reset = vkResetFences(adapter.device, 1u, &fence);
  if (reset != VK_SUCCESS) {
    SetVulkanLastError(adapter, VulkanFailureReason(
                                    reset, "accel_vulkan_command_unavailable"));
    return false;
  }
  const std::uint64_t ready_value = gate.next_ready_value++;
  std::atomic_thread_fence(std::memory_order_release);
  VkSemaphoreSignalInfoKHR signal{};
  signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR;
  signal.semaphore = gate.ready;
  signal.value = ready_value;
  const VkResult signaled = adapter.timeline.signal(adapter.device, &signal);
  if (signaled != VK_SUCCESS) {
    SetVulkanLastError(
        adapter, VulkanFailureReason(signaled, "accel_vulkan_timeline_failed"));
    return false;
  }
  const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkTimelineSemaphoreSubmitInfoKHR timeline{};
  timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR;
  timeline.waitSemaphoreValueCount = 1u;
  timeline.pWaitSemaphoreValues = &ready_value;
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.pNext = &timeline;
  submit.waitSemaphoreCount = 1u;
  submit.pWaitSemaphores = &gate.ready;
  submit.pWaitDstStageMask = &wait_stage;
  submit.commandBufferCount = static_cast<std::uint32_t>(commands.size());
  submit.pCommandBuffers = commands.data();
  const std::uint64_t submitted_ns =
      collect_timing ? MonotonicNanoseconds() : 0u;
  const VkResult submitted =
      vkQueueSubmit(adapter.compute_queue, 1u, &submit, fence);
  if (submitted != VK_SUCCESS) {
    SetVulkanLastError(
        adapter, VulkanFailureReason(submitted, "accel_vulkan_submit_failed"));
    return false;
  }
  const std::size_t tail =
      (adapter.pending_head + adapter.pending_size) % adapter.pending.size();
  adapter.pending[tail] = VulkanAdapter::Pending{
      .external_fence = fence,
      .completion = completion,
      .user = user,
      .submitted_ns = submitted_ns,
      .inflight = 1u,
      .timing = collect_timing,
      .timestamp = false,
      .external = true,
      .force_device_lost =
          adapter.device_loss_fault.take(SubmitKind::Work),
  };
  ++adapter.pending_size;
  RecordVulkanCommandSubmit(adapter);
  adapter.command_inflight_peak =
      std::max<std::uint64_t>(adapter.command_inflight_peak, 1u);
  completion_lock.unlock();
  adapter.completion_cv.notify_one();
  return true;
}

#endif

} // namespace rund::node::accel::detail::vulkan_sliding_detail
