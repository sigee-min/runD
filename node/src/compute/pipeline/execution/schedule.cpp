#include "schedule/local.hpp"

namespace rund::compute::detail {

Status prepare_pipeline_execution_schedule(
    const residency::execution::Plan &plan,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &pipelines,
    PipelineExecutionSchedulePrepared &prepared) noexcept {
  return prepare_pipeline_execution_schedule_impl(plan, pipelines, prepared);
}

Status submit_pipeline_execution_schedule(
    const residency::execution::Plan &plan,
    const PipelineExecutionSchedulePrepared &prepared,
    const residency::ExecutionLease &lease,
    const PipelineResidencyWindowReleaseCompletion release,
    const PipelineResidencyScheduleFinalCompletion final, void *const user,
    PipelineResidencyScheduleControl &control,
    node::accel::detail::PreparedResidencyStreamControl &stream) noexcept {
  return submit_pipeline_execution_schedule_impl(plan, prepared, lease, release,
                                                 final, user, control, stream);
}

Status
signal_pipeline_execution_schedule(PipelineResidencyScheduleControl &control,
                                   const std::uint64_t epoch,
                                   const Status admission) noexcept {
  return signal_pipeline_execution_schedule_impl(control, epoch, admission);
}

Status
abort_pipeline_execution_schedule(PipelineResidencyScheduleControl &control,
                                  const Status failure) noexcept {
  return abort_pipeline_execution_schedule_impl(control, failure);
}

} // namespace rund::compute::detail
