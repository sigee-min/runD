#include "internal.hpp"

#include <algorithm>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck
WaitVulkanTimelineDone(VulkanTimelineOwner &owner,
                       const VulkanTimelinePoint point,
                       const std::uint64_t timeout_ns) noexcept {
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  {
    std::lock_guard lock{owner.gate};
    if (!point || point.generation != owner.open_generation ||
        point.value > owner.generation_last_value ||
        point.value > owner.done_signal_scheduled || timeout_ns == 0u) {
      return timeline_owner::invalid();
    }
    const VulkanTimelinePoint marked = owner.terminal_loss;
    if (marked.generation == point.generation &&
        marked.value == point.value && marked.ready_value == point.ready_value &&
        marked.ready_cell == point.ready_cell) {
      owner.terminal_loss = {};
      return timeline_owner::failed();
    }
  }
  const std::uint64_t value = point.value;
  VkSemaphoreWaitInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR;
  info.semaphoreCount = 1u;
  info.pSemaphores = &owner.done;
  info.pValues = &value;
  return owner.wait(owner.device, &info, timeout_ns) == VK_SUCCESS
             ? rund::AccelCheck{true, "ok"}
             : timeline_owner::failed();
}

rund::AccelCheck ReadVulkanTimeline(const VulkanTimelineOwner &owner,
                                    std::uint64_t &ready,
                                    std::uint64_t &done) noexcept {
  ready = 0u;
  done = 0u;
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  for (const VkSemaphore semaphore : owner.ready) {
    std::uint64_t current = 0u;
    if (owner.counter(owner.device, semaphore, &current) != VK_SUCCESS) {
      ready = 0u;
      done = 0u;
      return timeline_owner::failed();
    }
    ready = std::max(ready, current);
  }
  if (owner.counter(owner.device, owner.done, &done) != VK_SUCCESS) {
    ready = 0u;
    done = 0u;
    return timeline_owner::failed();
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
