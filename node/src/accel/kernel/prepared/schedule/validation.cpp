#include "local.hpp"

#include "../model.hpp"

#include <algorithm>
#include <array>
#include <functional>

namespace rund::node::accel::detail {

[[nodiscard]] prepared::PipelineState *
PreparedScheduleState(const PreparedKernelPipeline &pipeline) noexcept {
  return static_cast<prepared::PipelineState *>(pipeline.owner.get());
}

[[nodiscard]] std::size_t PreparedScheduleSortedStates(
    const PreparedResidencyScheduleRequest &request,
    std::array<prepared::PipelineState *, ResidencyScheduleRoleCapacity>
        &states) noexcept {
  std::size_t count = 0u;
  for (std::size_t role = 0u; role < request.role_count; ++role) {
    prepared::PipelineState *const state =
        PreparedScheduleState(request.roles[role].pipeline);
    bool found = false;
    for (std::size_t prior = 0u; prior < count; ++prior) {
      found = found || states[prior] == state;
    }
    if (!found) {
      states[count++] = state;
    }
  }
  std::sort(states.begin(), states.begin() + count,
            std::less<prepared::PipelineState *>{});
  return count;
}

[[nodiscard]] bool PreparedScheduleValidRequest(
    const rund::AccelContext &context,
    const PreparedResidencyScheduleRequest &request, const BackendOps *&ops,
    std::array<prepared::PipelineState *, ResidencyScheduleRoleCapacity>
        &states,
    std::size_t &state_count) noexcept {
  ops = nullptr;
  state_count = 0u;
  if (request.plan_identity == 0u || request.token == 0u ||
      request.generation == 0u || request.epoch_count == 0u ||
      request.role_count != ResidencyScheduleRoleCapacity ||
      request.release == nullptr || request.final == nullptr ||
      request.user == nullptr) {
    return false;
  }
  for (std::size_t role = 0u; role < request.role_count; ++role) {
    const PreparedResidencyScheduleRole &value = request.roles[role];
    prepared::PipelineState *const state = PreparedScheduleState(value.pipeline);
    if (!value.pipeline.ok || state == nullptr || value.role != role ||
        value.bank != role % 2u || value.local_count == 0u ||
        value.first_control_generation == 0u ||
        value.control_generation_stride == 0u ||
        value.local_count > ResidencyWindowLocalCapacity ||
        !prepared::ValidPipeline(context, *state) || state->ops == nullptr ||
        state->backend == nullptr ||
        state->ops->prepare_prepared_schedule == nullptr ||
        state->ops->submit_prepared_schedule == nullptr ||
        state->ops->signal_prepared_schedule == nullptr ||
        state->ops->abort_prepared_schedule == nullptr ||
        (ops != nullptr && ops != state->ops)) {
      return false;
    }
    ops = state->ops;
  }
  const std::size_t tail_role =
      static_cast<std::size_t>((request.epoch_count - 1u) % request.role_count);
  if (request.tail_local_count == 0u ||
      request.tail_local_count > request.roles[tail_role].local_count) {
    return false;
  }
  state_count = PreparedScheduleSortedStates(request, states);
  return state_count != 0u;
}

} // namespace rund::node::accel::detail
