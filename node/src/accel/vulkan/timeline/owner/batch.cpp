#include "../../adapter/error.hpp"

#include "internal.hpp"

#include "../../adapter/state.hpp"
#include "../../local.hpp"
#include "../../runtime/counter.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace timeline_owner {

rund::AccelCheck SubmitVulkanTimelineBatches(
    VulkanAdapter &adapter, const std::span<const VulkanTimelineBatch> batches,
    std::uint64_t &queue_calls, const bool adapter_locked) noexcept {
  queue_calls = 0u;
  VulkanTimelineOwner &owner = adapter.timeline;
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  if (batches.empty() ||
      batches.size() > std::numeric_limits<std::uint32_t>::max()) {
    return invalid();
  }
  for (std::size_t index = 0u; index < batches.size(); ++index) {
    const VulkanTimelineBatch &batch = batches[index];
    if (!batch.point || batch.commands.empty() ||
        batch.commands.size() > std::numeric_limits<std::uint32_t>::max() ||
        std::any_of(batch.commands.begin(), batch.commands.end(),
                    [](const VkCommandBuffer command) {
                      return command == VK_NULL_HANDLE;
                    }) ||
        (index != 0u &&
         (batch.point.generation != batches.front().point.generation ||
          batch.point.value != batches[index - 1u].point.value + 1u))) {
      return invalid();
    }
  }

  std::unique_lock adapter_lock{adapter.mutex, std::defer_lock};
  if (!adapter_locked) {
    adapter_lock.lock();
  }
  std::lock_guard owner_lock{owner.gate};
  const VulkanTimelinePoint first = batches.front().point;
  const VulkanTimelinePoint last = batches.back().point;
  if (first.generation != owner.open_generation ||
      last.value > owner.generation_last_value ||
      first.value != owner.done_signal_scheduled + 1u ||
      !within_difference(owner, owner.done, last.value, true)) {
    return invalid();
  }
  std::array<std::uint64_t, VulkanTimelineWindowCapacity> scheduled =
      owner.ready_wait_scheduled;
  for (const VulkanTimelineBatch &batch : batches) {
    const std::size_t cell = batch.point.ready_cell;
    if (scheduled[cell] == std::numeric_limits<std::uint64_t>::max() ||
        batch.point.ready_value != scheduled[cell] + 1u ||
        !within_difference(owner, owner.ready[cell], batch.point.ready_value,
                           false)) {
      return invalid();
    }
    scheduled[cell] = batch.point.ready_value;
  }

  std::vector<std::uint64_t> ready_values{};
  std::vector<std::uint64_t> done_values{};
  std::vector<VkPipelineStageFlags> stages{};
  std::vector<VkTimelineSemaphoreSubmitInfoKHR> timelines{};
  std::vector<VkSubmitInfo> submits{};
  try {
    ready_values.resize(batches.size());
    done_values.resize(batches.size());
    stages.resize(batches.size());
    timelines.resize(batches.size());
    submits.resize(batches.size());
  } catch (...) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  for (std::size_t index = 0u; index < batches.size(); ++index) {
    ready_values[index] = batches[index].point.ready_value;
    done_values[index] = batches[index].point.value;
    stages[index] = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    timelines[index] = VkTimelineSemaphoreSubmitInfoKHR{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
        .waitSemaphoreValueCount = 1u,
        .pWaitSemaphoreValues = &ready_values[index],
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &done_values[index],
    };
    submits[index] = VkSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timelines[index],
        .waitSemaphoreCount = 1u,
        .pWaitSemaphores = &owner.ready[batches[index].point.ready_cell],
        .pWaitDstStageMask = &stages[index],
        .commandBufferCount =
            static_cast<std::uint32_t>(batches[index].commands.size()),
        .pCommandBuffers = batches[index].commands.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &owner.done,
    };
  }
  const VkResult submitted =
      adapter.device_loss_fault.take(SubmitKind::Work)
          ? VK_ERROR_DEVICE_LOST
          : vkQueueSubmit(adapter.compute_queue,
                          static_cast<std::uint32_t>(batches.size()),
                          submits.data(), VK_NULL_HANDLE);
  if (submitted != VK_SUCCESS) {
    const char *const reason =
        VulkanFailureReason(submitted, "accel_vulkan_submit_failed");
    SetVulkanLastError(adapter, reason);
    if (submitted == VK_ERROR_DEVICE_LOST) {
      adapter.residency_quarantined.store(true, std::memory_order_release);
    }
    return rund::AccelCheck{false, reason};
  }
  if (adapter.fault_residency_terminal_once.exchange(
          false, std::memory_order_acq_rel)) {
    owner.terminal_loss = last;
  }
  owner.ready_wait_scheduled = scheduled;
  owner.done_signal_scheduled = last.value;
  owner.window_ready_mode = true;
  RecordVulkanCommandSubmit(adapter);
  adapter.command_inflight_peak =
      std::max<std::uint64_t>(adapter.command_inflight_peak, 1u);
  queue_calls = 1u;
  return rund::AccelCheck{true, "ok"};
}

} // namespace timeline_owner

rund::AccelCheck SubmitVulkanTimelineWindow(
    VulkanAdapter &adapter, const std::span<const VulkanTimelineBatch> batches,
    std::uint64_t &queue_calls, const bool adapter_locked) noexcept {
  queue_calls = 0u;
  if (batches.size() > VulkanTimelineWindowCapacity) {
    return timeline_owner::invalid();
  }
  return timeline_owner::SubmitVulkanTimelineBatches(adapter, batches,
                                                     queue_calls,
                                                     adapter_locked);
}

rund::AccelCheck SubmitVulkanTimelineStream(
    VulkanAdapter &adapter, const std::span<const VulkanTimelineBatch> batches,
    std::uint64_t &queue_calls, const bool adapter_locked) noexcept {
  return timeline_owner::SubmitVulkanTimelineBatches(adapter, batches,
                                                     queue_calls,
                                                     adapter_locked);
}

#endif

} // namespace rund::node::accel::detail
