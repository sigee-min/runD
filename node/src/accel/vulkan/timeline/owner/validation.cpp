#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace timeline_owner {

[[nodiscard]] rund::AccelCheck unsupported() noexcept {
  return rund::AccelCheck{false, "compute_backend_unsupported"};
}

[[nodiscard]] rund::AccelCheck invalid() noexcept {
  return rund::AccelCheck{false, "accel_vulkan_timeline_invalid"};
}

[[nodiscard]] rund::AccelCheck failed() noexcept {
  return rund::AccelCheck{false, "accel_vulkan_timeline_failed"};
}

void record_failure(VulkanTimelineOwner &owner,
                    const rund::AccelCheck check) noexcept {
  owner.init_state = VulkanTimelineInitState::Failed;
  owner.init_check = check;
}

[[nodiscard]] PFN_vkVoidFunction load(const VkDevice device,
                                      const VulkanTimelineRoute route,
                                      const char *const core,
                                      const char *const extension) noexcept {
  if (device == VK_NULL_HANDLE) {
    return nullptr;
  }
  const char *const name =
      route == VulkanTimelineRoute::Core ? core : extension;
  return vkGetDeviceProcAddr(device, name);
}

[[nodiscard]] bool valid(const VulkanTimelineOwner &owner) noexcept {
  return owner.init_state == VulkanTimelineInitState::Initialized &&
         owner.init_check.ok && owner.support &&
         owner.support.max_difference != 0u && owner.device != VK_NULL_HANDLE &&
         std::all_of(
             owner.ready.begin(), owner.ready.end(),
             [](const VkSemaphore ready) { return ready != VK_NULL_HANDLE; }) &&
         owner.done != VK_NULL_HANDLE && owner.signal != nullptr &&
         owner.wait != nullptr && owner.counter != nullptr;
}

[[nodiscard]] bool within_difference(const VulkanTimelineOwner &owner,
                                     const VkSemaphore semaphore,
                                     const std::uint64_t value,
                                     const bool strictly_future) noexcept {
  std::uint64_t current = 0u;
  if (owner.support.max_difference == 0u ||
      owner.counter(owner.device, semaphore, &current) != VK_SUCCESS ||
      value < current || (strictly_future && value == current)) {
    return false;
  }
  return value - current <= owner.support.max_difference;
}

[[nodiscard]] bool
valid_ready_signal(const VulkanTimelineOwner &owner,
                   const VulkanTimelinePoint point) noexcept {
  const std::size_t ready_cell = point.ready_cell;
  if (!point || point.generation != owner.open_generation ||
      ready_cell >= owner.ready_signaled.size()) {
    return false;
  }
  const std::uint64_t ready_value = point.ready_value;
  return owner.ready_signaled[ready_cell] !=
             std::numeric_limits<std::uint64_t>::max() &&
         ready_value == owner.ready_signaled[ready_cell] + 1u &&
         ready_value <= owner.ready_wait_scheduled[ready_cell] &&
         within_difference(owner, owner.ready[ready_cell], ready_value, true);
}

} // namespace timeline_owner

#endif

} // namespace rund::node::accel::detail
