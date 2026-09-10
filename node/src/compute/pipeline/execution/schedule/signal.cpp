#include "local.hpp"

#include "../../../backend.hpp"
#include "../../../status.hpp"
#include "../../state.hpp"

#include <mutex>
#include <span>

namespace rund::compute::detail {

Status signal_pipeline_execution_schedule_impl(
    PipelineResidencyScheduleControl &control, const std::uint64_t epoch,
    Status admission) noexcept {
  std::shared_ptr<PipelineState> pipeline{};
  node::accel::detail::PreparedKernelPipeline prepared{};
  node::accel::detail::BackendResidencyWindowSignal signal{};
  {
    std::lock_guard lock{control.gate};
    if (!control.active || control.plan == nullptr ||
        epoch >= control.lease.epochs) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t role = static_cast<std::size_t>(
        epoch % node::accel::detail::ResidencyScheduleRoleCapacity);
    PipelineExecutionAttempt &attempt = control.attempts[role];
    if (attempt.started || attempt.signalled) {
      return Status::fail(Reason::PipelineBusy);
    }
    pipeline = control.pipelines[role];
    if (pipeline == nullptr || pipeline->device == nullptr ||
        pipeline->device->ops == nullptr ||
        pipeline->device->ops->residency.signal_residency_schedule == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    residency::execution::Node dispatch{};
    std::uint32_t generation = 0u;
    const auto &role_plan = control.native.bound.roles[role];
    if (!control.plan->project(
            residency::execution::NodeId{
                .epoch = epoch, .phase = residency::execution::Phase::Dispatch},
            dispatch) ||
        !pipeline_schedule_generation_for(role_plan, epoch, generation)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    signal = node::accel::detail::BackendResidencyWindowSignal{
        .admission =
            admission
                ? rund::AccelCheck{true, "ok"}
                : rund::AccelCheck{false,
                                   reason_message(admission.reason()).data()},
        .plan_identity = control.lease.plan,
        .token = control.lease.token,
        .generation = control.lease.generation,
        .epoch = epoch,
        .control_generation = generation,
        .bank = static_cast<std::uint8_t>(dispatch.bank),
    };
    prepared = role_plan.pipeline;
    if (admission) {
      const std::size_t local_count =
          epoch + 1u == control.lease.epochs
              ? control.native.bound.tail_local_count
              : role_plan.local_count;
      const Status started = begin_pipeline_execution_attempt(
          *pipeline, dispatch,
          std::span<const std::uint32_t>{role_plan.locals.data(), local_count},
          generation, attempt);
      if (!started) {
        admission = started;
        signal.admission = {false, reason_message(started.reason()).data()};
      }
    }
    attempt.signalled = true;
  }
  Status signalled = pipeline->device->ops->residency.signal_residency_schedule(
      *pipeline->device, prepared, signal);
  if (!signalled && signal.admission.ok) {
    {
      std::lock_guard lock{control.gate};
      const std::size_t role = static_cast<std::size_t>(
          epoch % node::accel::detail::ResidencyScheduleRoleCapacity);
      PipelineExecutionAttempt &attempt = control.attempts[role];
      reject_pipeline_execution_attempt(*pipeline, attempt, signalled);
      attempt.signalled = true;
    }
    signal.admission = {false, reason_message(signalled.reason()).data()};
    signalled = pipeline->device->ops->residency.signal_residency_schedule(
        *pipeline->device, prepared, signal);
  }
  if (!signalled) {
    std::lock_guard lock{control.gate};
    control.attempts[epoch % node::accel::detail::ResidencyScheduleRoleCapacity]
        .signalled = false;
  }
  return signalled;
}

} // namespace rund::compute::detail
