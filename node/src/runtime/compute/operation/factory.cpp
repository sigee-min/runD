#include "internal.hpp"

#include <utility>

namespace rund::node::compute_operation_detail {
namespace {

void ReleaseOwner(std::shared_ptr<void> &owner) noexcept { owner.reset(); }

} // namespace

const compute_detail::OperationTable JobOperationTable{
    .backend = JobBackend,
    .workers = JobWorkers,
    .reserve = ReserveJob,
    .submit_cpu = SubmitJobCpu,
    .advance_cpu = AdvanceJobCpu,
    .result_cpu = ResultJobCpu,
    .submit_accel = SubmitJobAccel,
    .result_accel = ResultJobAccel,
    .fail = FailJob,
    .cancel = CancelJob,
    .record_frame = RecordJobFrame,
    .release = ReleaseOwner,
};

const compute_detail::OperationTable PipelineOperationTable{
    .backend = PipelineBackend,
    .workers = PipelineWorkers,
    .reserve = ReservePipeline,
    .submit_cpu = SubmitPipelineCpu,
    .advance_cpu = AdvancePipelineCpu,
    .result_cpu = ResultPipelineCpu,
    .submit_accel = SubmitPipelineAccel,
    .result_accel = ResultPipelineAccel,
    .fail = FailPipeline,
    .cancel = CancelPipeline,
    .record_frame = RecordPipelineFrame,
    .release = ReleaseOwner,
};

const compute_detail::OperationTable VirtualOperationTable{
    .backend = VirtualBackend,
    .workers = VirtualWorkers,
    .reserve = ReserveVirtual,
    .submit_cpu = SubmitVirtual,
    .advance_cpu = AdvanceVirtual,
    .result_cpu = ResultVirtual,
    .submit_accel = SubmitVirtual,
    .resume_accel = AdvanceVirtual,
    .result_accel = ResultVirtual,
    .fail = FailVirtual,
    .cancel = CancelVirtual,
    .release = ReleaseOwner,
};

} // namespace rund::node::compute_operation_detail

namespace rund::node {

compute_detail::Operation compute_detail::make_job(
    std::shared_ptr<compute::detail::JobState> state) noexcept {
  return Operation{.table = &compute_operation_detail::JobOperationTable,
                   .owner = std::move(state)};
}

compute_detail::Operation compute_detail::make_pipeline(
    std::shared_ptr<compute::detail::PipelineState> state) noexcept {
  return Operation{.table = &compute_operation_detail::PipelineOperationTable,
                   .owner = std::move(state)};
}

compute_detail::Operation compute_detail::make_virtual_pipeline(
    std::shared_ptr<compute::detail::VirtualPipelineState> state) noexcept {
  return Operation{.table = &compute_operation_detail::VirtualOperationTable,
                   .owner = std::move(state)};
}

compute_detail::Operation
compute_detail::make_operation(std::shared_ptr<void> owner,
                               const void *const table) noexcept {
  return Operation{.table = static_cast<const OperationTable *>(table),
                   .owner = std::move(owner)};
}

} // namespace rund::node
