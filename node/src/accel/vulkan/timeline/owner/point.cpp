#include "internal.hpp"

#include <limits>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck
ReserveVulkanTimelinePoint(VulkanTimelineOwner &owner,
                           const std::uint32_t generation,
                           VulkanTimelinePoint &point) noexcept {
  point = {};
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  std::lock_guard lock{owner.gate};
  if (generation == 0u || generation != owner.open_generation) {
    return timeline_owner::invalid();
  }
  if (owner.next_value == 0u ||
      owner.next_value == std::numeric_limits<std::uint64_t>::max()) {
    return rund::AccelCheck{false, "accel_vulkan_timeline_capacity"};
  }
  const std::uint64_t cell = (owner.next_value - owner.generation_first_value) %
                             VulkanTimelineWindowCapacity;
  if (owner.next_ready_value[cell] == 0u ||
      owner.next_ready_value[cell] ==
          std::numeric_limits<std::uint64_t>::max()) {
    return rund::AccelCheck{false, "accel_vulkan_timeline_capacity"};
  }
  point = VulkanTimelinePoint{.generation = generation,
                              .value = owner.next_value++,
                              .ready_value = owner.next_ready_value[cell]++,
                              .ready_cell = static_cast<std::uint8_t>(cell)};
  owner.generation_ready_value[cell] = point.ready_value;
  owner.generation_last_value = point.value;
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
CanSignalVulkanTimelineReady(VulkanTimelineOwner &owner,
                             const VulkanTimelinePoint point) noexcept {
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  std::lock_guard owner_lock{owner.gate};
  if (!timeline_owner::valid_ready_signal(owner, point)) {
    return timeline_owner::invalid();
  }
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
SignalVulkanTimelineReady(VulkanTimelineOwner &owner,
                          const VulkanTimelinePoint point) noexcept {
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  std::lock_guard owner_lock{owner.gate};
  const std::uint64_t ready_value = point.ready_value;
  const std::size_t ready_cell = point.ready_cell;
  if (!timeline_owner::valid_ready_signal(owner, point)) {
    return timeline_owner::invalid();
  }
  VkSemaphoreSignalInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR;
  info.semaphore = owner.ready[ready_cell];
  info.value = ready_value;
  if (owner.signal(owner.device, &info) != VK_SUCCESS) {
    return timeline_owner::failed();
  }
  owner.ready_signaled[ready_cell] = ready_value;
  if (!owner.window_ready_mode) {
    owner.ordered_ready_signaled = ready_value;
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
