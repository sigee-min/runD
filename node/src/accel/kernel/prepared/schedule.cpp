#include "schedule/local.hpp"

#include "model.hpp"

#include <array>
#include <mutex>

namespace rund::node::accel::detail {

rund::AccelCheck SubmitPreparedKernelPipelineSchedule(
    const rund::AccelContext &context,
    const PreparedResidencyScheduleRequest &request,
    PreparedResidencyScheduleControl &control,
    PreparedResidencyStreamControl &stream) noexcept {
  const BackendOps *ops = nullptr;
  std::array<prepared::PipelineState *, ResidencyScheduleRoleCapacity> states{};
  std::size_t state_count = 0u;
  if (!PreparedScheduleValidRequest(context, request, ops, states, state_count)) {
    return {false, "accel_kernel_run_invalid"};
  }
  std::array<std::unique_lock<std::mutex>, ResidencyScheduleRoleCapacity>
      claims{};
  for (std::size_t index = 0u; index < state_count; ++index) {
    claims[index] =
        std::unique_lock<std::mutex>{states[index]->submission.mutex};
  }
  std::scoped_lock control_lock{stream.gate, control.gate};
  if (!stream.active || stream.quarantined || control.active ||
      request.plan_identity != stream.plan_identity ||
      request.token != stream.token ||
      request.generation != stream.generation) {
    return {false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < state_count; ++index) {
    const prepared::PipelineSubmission &submission = states[index]->submission;
    if (submission.stream != &stream || submission.window != nullptr ||
        submission.schedule != nullptr || submission.quarantined ||
        submission.owner == nullptr) {
      return {false, "compute_pipeline_busy"};
    }
  }
  control.bound = request;
  control.backend = BackendResidencyScheduleRequest{
      .plan_identity = request.plan_identity,
      .token = request.token,
      .generation = request.generation,
      .epoch_count = request.epoch_count,
      .role_count = request.role_count,
      .tail_local_count = request.tail_local_count,
      .lowering = request.lowering,
      .admission = request.admission,
      .release = CompletePreparedScheduleRelease,
      .final = CompletePreparedScheduleFinal,
      .user = &control,
  };
  for (std::size_t role = 0u; role < request.role_count; ++role) {
    const PreparedResidencyScheduleRole &source = request.roles[role];
    prepared::PipelineState *const state = PreparedScheduleState(source.pipeline);
    control.backend.roles[role] = BackendResidencyScheduleRole{
        .prepared = state->backend,
        .locals = source.locals,
        .local_count = source.local_count,
        .first_control_generation = source.first_control_generation,
        .control_generation_stride = source.control_generation_stride,
        .role = source.role,
        .bank = source.bank,
    };
  }
  control.states = states;
  control.stream = &stream;
  control.release_count = 0u;
  control.success_prefix = 0u;
  control.suppressed_first = 0u;
  control.suppressed_count = 0u;
  control.first_failure = {true, "ok"};
  control.active = true;
  control.aborting = false;
  control.malformed = false;
  control.quarantined = false;
  for (std::size_t index = 0u; index < state_count; ++index) {
    states[index]->submission.schedule = &control;
  }
  const rund::AccelCheck submitted =
      ops->submit_prepared_schedule(control.backend);
  if (submitted.ok) {
    return submitted;
  }
  for (std::size_t index = 0u; index < state_count; ++index) {
    if (states[index]->submission.schedule == &control) {
      states[index]->submission.schedule = nullptr;
    }
  }
  control.bound = {};
  control.backend = {};
  control.states.fill(nullptr);
  control.stream = nullptr;
  control.active = false;
  return submitted;
}


} // namespace rund::node::accel::detail
