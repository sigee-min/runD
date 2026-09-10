#include "local.hpp"

#include "../model.hpp"

#include <array>

namespace rund::node::accel::detail {

BackendResidencySchedulePreparation PrepareKernelPipelineSchedule(
    const rund::AccelContext &context,
    const std::span<const PreparedResidencyScheduleRole> roles,
    const std::uint64_t epoch_count,
    const std::size_t tail_local_count) noexcept {
  if (roles.size() != ResidencyScheduleRoleCapacity || epoch_count == 0u ||
      tail_local_count == 0u ||
      tail_local_count > roles[(epoch_count - 1u) % roles.size()].local_count) {
    return {};
  }
  std::array<BackendResidencyScheduleRole, ResidencyScheduleRoleCapacity>
      backend{};
  const BackendOps *ops = nullptr;
  for (std::size_t role = 0u; role < roles.size(); ++role) {
    const PreparedResidencyScheduleRole &value = roles[role];
    prepared::PipelineState *const state = PreparedScheduleState(value.pipeline);
    if (!value.pipeline.ok || state == nullptr || value.role != role ||
        value.bank != role % 2u || value.local_count == 0u ||
        value.first_control_generation == 0u ||
        value.control_generation_stride == 0u ||
        value.local_count > ResidencyWindowLocalCapacity ||
        !prepared::ValidPipeline(context, *state) || state->ops == nullptr ||
        state->ops->prepare_prepared_schedule == nullptr ||
        (ops != nullptr && ops != state->ops)) {
      return {};
    }
    ops = state->ops;
    backend[role] = BackendResidencyScheduleRole{
        .prepared = state->backend,
        .locals = value.locals,
        .local_count = value.local_count,
        .first_control_generation = value.first_control_generation,
        .control_generation_stride = value.control_generation_stride,
        .role = value.role,
        .bank = value.bank,
    };
  }
  return ops->prepare_prepared_schedule(backend, epoch_count, tail_local_count);
}

} // namespace rund::node::accel::detail
