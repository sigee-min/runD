#include "internal.hpp"

#include "../../../../timeline/owner.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace schedule {

rund::AccelCheck Invalid() noexcept {
  return {false, "accel_kernel_pipeline_invalid"};
}

rund::AccelCheck Unavailable() noexcept {
  return {false, "accel_vulkan_command_unavailable"};
}

bool UniqueLocals(const BackendResidencyScheduleRole &role,
                  const std::size_t count) noexcept {
  if (count == 0u || count > role.local_count ||
      role.local_count > ResidencyWindowLocalCapacity) {
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (role.locals[index] == role.locals[prior]) {
        return false;
      }
    }
  }
  return true;
}

bool Add(const std::uint64_t left, const std::uint64_t right,
         std::uint64_t &value) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  value = left + right;
  return true;
}

bool ControlGeneration(const BackendResidencyScheduleRole &role,
                       const std::uint64_t epoch,
                       std::uint32_t &generation) noexcept {
  const std::uint64_t turn = epoch / ResidencyScheduleRoleCapacity;
  const std::uint64_t delta =
      turn * static_cast<std::uint64_t>(role.control_generation_stride);
  const std::uint64_t value =
      static_cast<std::uint64_t>(role.first_control_generation) + delta;
  if (turn != 0u && delta / turn != static_cast<std::uint64_t>(
                                        role.control_generation_stride)) {
    return false;
  }
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  generation = static_cast<std::uint32_t>(value);
  return true;
}

VulkanTimelinePoint Point(const VulkanResidencyScheduleRun &run,
                          const std::uint64_t epoch) noexcept {
  const std::size_t cell =
      static_cast<std::size_t>(epoch % ResidencyScheduleRoleCapacity);
  const std::uint64_t turn = epoch / ResidencyScheduleRoleCapacity;
  return VulkanTimelinePoint{
      .generation = run.timeline_generation,
      .value = run.done_base + epoch,
      .ready_value = run.ready_base[cell] + turn,
      .ready_cell = static_cast<std::uint8_t>(cell),
  };
}

VulkanResidencyScheduleRole &NativeRole(VulkanResidencyScheduleRun &run,
                                        const std::uint64_t epoch) noexcept {
  return epoch + 1u == run.request.epoch_count
             ? run.tail
             : run.roles[static_cast<std::size_t>(
                   epoch % ResidencyScheduleRoleCapacity)];
}

const BackendResidencyScheduleRole &
SourceRole(const VulkanResidencyScheduleRun &run,
           const std::uint64_t epoch) noexcept {
  return run.request
      .roles[static_cast<std::size_t>(epoch % ResidencyScheduleRoleCapacity)];
}

std::size_t LocalCount(const VulkanResidencyScheduleRun &run,
                       const std::uint64_t epoch) noexcept {
  return epoch + 1u == run.request.epoch_count
             ? run.request.tail_local_count
             : SourceRole(run, epoch).local_count;
}

} // namespace schedule

using namespace schedule;

BackendResidencySchedulePreparation PrepareVulkanResidencySchedule(
    const std::span<const BackendResidencyScheduleRole> roles,
    const std::uint64_t epoch_count,
    const std::size_t tail_local_count) noexcept {
  if (roles.size() != ResidencyScheduleRoleCapacity || epoch_count == 0u ||
      tail_local_count == 0u ||
      tail_local_count > roles[(epoch_count - 1u) % roles.size()].local_count) {
    return {};
  }
  VulkanAdapter *adapter = nullptr;
  for (std::size_t index = 0u; index < roles.size(); ++index) {
    const BackendResidencyScheduleRole &role = roles[index];
    auto *const pipeline = static_cast<VulkanPipeline *>(role.prepared.get());
    if (role.role != index || role.bank != index % 2u ||
        role.first_control_generation == 0u ||
        role.control_generation_stride == 0u ||
        !UniqueLocals(role, role.local_count) ||
        !ValidVulkanPipeline(pipeline) || pipeline->residency == nullptr ||
        !pipeline->residency->ready || !pipeline->residency->bounded ||
        (adapter != nullptr && adapter != pipeline->adapter)) {
      return {};
    }
    adapter = pipeline->adapter;
  }
  constexpr std::uint64_t per_epoch =
      sizeof(VulkanTimelineBatch) + sizeof(std::uint64_t) * 2u +
      sizeof(VkPipelineStageFlags) + sizeof(VkTimelineSemaphoreSubmitInfoKHR) +
      sizeof(VkSubmitInfo);
  std::uint64_t transient = 0u;
  if (epoch_count > std::numeric_limits<std::uint64_t>::max() / per_epoch ||
      !Add(0u, epoch_count * per_epoch, transient)) {
    return BackendResidencySchedulePreparation{
        .check = {false, "compute_pipeline_capacity"}};
  }
  return BackendResidencySchedulePreparation{
      .check = {true, "ok"},
      .kind = BackendResidencyScheduleLowering::VulkanTimeline,
      // The fixed four-role run is already part of the cold prepared owner.
      // Only the Q VkSubmitInfo lowering below is new and transient.
      .retained_bytes = 0u,
      .transient_bytes = transient,
      .queue_calls = 1u,
      .callbacks_async = true,
  };
}

#endif

} // namespace rund::node::accel::detail
