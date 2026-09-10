#include "internal.hpp"

#include <limits>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck
VulkanTimelineCapability(const VulkanTimelineOwner &owner) noexcept {
  if (owner.init_state != VulkanTimelineInitState::Initialized ||
      !owner.init_check.ok) {
    return owner.init_check;
  }
  return timeline_owner::valid(owner) ? rund::AccelCheck{true, "ok"}
                                      : timeline_owner::failed();
}

rund::AccelCheck
ReserveVulkanTimelineGeneration(VulkanTimelineOwner &owner,
                                std::uint32_t &generation) noexcept {
  generation = 0u;
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  std::lock_guard lock{owner.gate};
  if (owner.open_generation != 0u) {
    return timeline_owner::invalid();
  }
  if (owner.next_generation == 0u ||
      owner.next_generation == std::numeric_limits<std::uint32_t>::max()) {
    return rund::AccelCheck{false, "accel_vulkan_timeline_capacity"};
  }
  generation = owner.next_generation++;
  owner.open_generation = generation;
  owner.generation_ready_value.fill(0u);
  owner.generation_first_ready_value = owner.next_ready_value;
  owner.generation_first_value = owner.next_value;
  owner.generation_last_value = 0u;
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
CancelVulkanTimelineGeneration(VulkanTimelineOwner &owner,
                               const std::uint32_t generation) noexcept {
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  std::lock_guard lock{owner.gate};
  if (generation == 0u || generation != owner.open_generation ||
      owner.generation_first_value == 0u ||
      owner.done_signal_scheduled >= owner.generation_first_value ||
      owner.next_generation != generation + 1u) {
    return timeline_owner::invalid();
  }
  for (std::size_t cell = 0u; cell < owner.generation_ready_value.size();
       ++cell) {
    const std::uint64_t first = owner.generation_first_ready_value[cell];
    const std::uint64_t last = owner.generation_ready_value[cell];
    if (first == 0u ||
        (last != 0u && (owner.ready_signaled[cell] >= first ||
                        owner.ready_wait_scheduled[cell] >= first ||
                        last == std::numeric_limits<std::uint64_t>::max() ||
                        owner.next_ready_value[cell] != last + 1u))) {
      return timeline_owner::invalid();
    }
  }
  owner.next_value = owner.generation_first_value;
  for (std::size_t cell = 0u; cell < owner.generation_ready_value.size();
       ++cell) {
    owner.next_ready_value[cell] = owner.generation_first_ready_value[cell];
  }
  owner.generation_ready_value.fill(0u);
  owner.generation_first_ready_value.fill(0u);
  owner.next_generation = generation;
  owner.open_generation = 0u;
  owner.generation_first_value = 0u;
  owner.generation_last_value = 0u;
  owner.window_ready_mode = false;
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
CloseVulkanTimelineGeneration(VulkanTimelineOwner &owner,
                              const VulkanTimelinePoint terminal) noexcept {
  const rund::AccelCheck capability = VulkanTimelineCapability(owner);
  if (!capability.ok) {
    return capability;
  }
  std::lock_guard lock{owner.gate};
  if (!terminal || terminal.generation != owner.open_generation ||
      terminal.value != owner.generation_last_value ||
      terminal.value > owner.done_signal_scheduled) {
    return timeline_owner::invalid();
  }
  std::uint64_t current = 0u;
  if (owner.counter(owner.device, owner.done, &current) != VK_SUCCESS) {
    return timeline_owner::failed();
  }
  if (current < terminal.value) {
    return timeline_owner::invalid();
  }
  owner.open_generation = 0u;
  owner.generation_first_value = 0u;
  owner.generation_last_value = 0u;
  owner.ordered_ready_signaled = owner.ready_signaled[0u];
  owner.generation_ready_value.fill(0u);
  owner.generation_first_ready_value.fill(0u);
  owner.window_ready_mode = false;
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
