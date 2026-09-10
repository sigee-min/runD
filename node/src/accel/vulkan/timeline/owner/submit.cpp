#include "internal.hpp"

#include "../../adapter/state.hpp"

#include <algorithm>
#include <limits>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck
SubmitVulkanTimeline(VulkanAdapter &adapter,
                     const std::span<const VkCommandBuffer> commands,
                     const VulkanTimelinePoint point) noexcept {
  VulkanTimelineOwner &owner = adapter.timeline;
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  if (!point || commands.size() > std::numeric_limits<std::uint32_t>::max() ||
      std::any_of(commands.begin(), commands.end(),
                  [](const VkCommandBuffer command) {
                    return command == VK_NULL_HANDLE;
                  })) {
    return timeline_owner::invalid();
  }
  std::scoped_lock lock{adapter.mutex, owner.gate};
  const std::uint64_t ready_value = point.ready_value;
  const std::uint64_t done_value = point.value;
  const std::size_t cell = point.ready_cell;
  if (point.generation != owner.open_generation || owner.window_ready_mode ||
      owner.ready_wait_scheduled[cell] ==
          std::numeric_limits<std::uint64_t>::max() ||
      owner.done_signal_scheduled ==
          std::numeric_limits<std::uint64_t>::max() ||
      ready_value <= owner.ready_wait_scheduled[cell] ||
      done_value != owner.done_signal_scheduled + 1u ||
      !timeline_owner::within_difference(owner, owner.ready[cell],
                                         ready_value, false) ||
      !timeline_owner::within_difference(owner, owner.done, done_value, true)) {
    return timeline_owner::invalid();
  }
  const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkTimelineSemaphoreSubmitInfoKHR timeline{};
  timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR;
  timeline.waitSemaphoreValueCount = 1u;
  timeline.pWaitSemaphoreValues = &ready_value;
  timeline.signalSemaphoreValueCount = 1u;
  timeline.pSignalSemaphoreValues = &done_value;
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.pNext = &timeline;
  submit.waitSemaphoreCount = 1u;
  submit.pWaitSemaphores = &owner.ready[cell];
  submit.pWaitDstStageMask = &wait_stage;
  submit.commandBufferCount = static_cast<std::uint32_t>(commands.size());
  submit.pCommandBuffers = commands.data();
  submit.signalSemaphoreCount = 1u;
  submit.pSignalSemaphores = &owner.done;
  if (vkQueueSubmit(adapter.compute_queue, 1u, &submit, VK_NULL_HANDLE) !=
      VK_SUCCESS) {
    return timeline_owner::failed();
  }
  owner.ready_wait_scheduled[cell] = ready_value;
  owner.done_signal_scheduled = done_value;
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
