#include "local.hpp"

#include "../model.hpp"

#include <memory>
#include <mutex>

namespace rund::node::accel::detail {

rund::AccelCheck SignalPreparedKernelPipelineSchedule(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    const BackendResidencyWindowSignal &signal) noexcept {
  prepared::PipelineState *const state = PreparedScheduleState(prepared);
  if (!prepared.ok || state == nullptr ||
      !prepared::ValidPipeline(context, *state) || state->ops == nullptr ||
      state->ops->signal_prepared_schedule == nullptr ||
      state->backend == nullptr || signal.control_generation == 0u) {
    return {false, "accel_kernel_run_invalid"};
  }
  PreparedResidencyScheduleControl *control = nullptr;
  {
    std::lock_guard lock{state->submission.mutex};
    control = static_cast<PreparedResidencyScheduleControl *>(
        state->submission.schedule);
  }
  if (control == nullptr) {
    return {false, "accel_kernel_run_invalid"};
  }
  {
    std::lock_guard lock{control->gate};
    if (!control->active || control->quarantined || control->aborting ||
        signal.plan_identity != control->bound.plan_identity ||
        signal.token != control->bound.token ||
        signal.generation != control->bound.generation ||
        signal.epoch >= control->bound.epoch_count) {
      return {false, "accel_kernel_run_invalid"};
    }
    const std::size_t role =
        static_cast<std::size_t>(signal.epoch % control->bound.role_count);
    const PreparedResidencyScheduleRole &expected = control->bound.roles[role];
    if (expected.pipeline.owner != prepared.owner ||
        expected.bank != signal.bank) {
      return {false, "accel_kernel_run_invalid"};
    }
  }
  return state->ops->signal_prepared_schedule(state->backend, signal);
}

rund::AccelCheck AbortPreparedKernelPipelineSchedule(
    const rund::AccelContext &context,
    PreparedResidencyScheduleControl &control,
    const BackendResidencyWindowAbort &abort) noexcept {
  const BackendOps *ops = nullptr;
  std::shared_ptr<void> leader{};
  {
    std::lock_guard lock{control.gate};
    if (!control.active || control.quarantined || control.aborting ||
        abort.failure.ok ||
        abort.plan_identity != control.bound.plan_identity ||
        abort.token != control.bound.token ||
        abort.generation != control.bound.generation) {
      return {false, "accel_kernel_run_invalid"};
    }
    const PreparedKernelPipeline &prepared = control.bound.roles[0u].pipeline;
    prepared::PipelineState *const state = PreparedScheduleState(prepared);
    if (state == nullptr || !prepared::ValidPipeline(context, *state) ||
        state->ops == nullptr ||
        state->ops->abort_prepared_schedule == nullptr) {
      return {false, "accel_kernel_run_invalid"};
    }
    ops = state->ops;
    leader = state->backend;
    control.aborting = true;
  }
  const rund::AccelCheck result = ops->abort_prepared_schedule(leader, abort);
  if (!result.ok) {
    std::lock_guard lock{control.gate};
    if (control.active) {
      control.aborting = false;
    }
  }
  return result;
}

} // namespace rund::node::accel::detail
