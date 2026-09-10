#pragma once

#include "../schedule.hpp"

namespace rund::compute::detail {

[[nodiscard]] bool pipeline_schedule_generation_for(
    const node::accel::detail::PreparedResidencyScheduleRole &,
    std::uint64_t epoch, std::uint32_t &generation) noexcept;

[[nodiscard]] Status build_pipeline_execution_schedule(
    const residency::execution::Plan &,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &,
    PipelineExecutionSchedulePrepared &, bool lower_backend) noexcept;

[[nodiscard]] Status prepare_pipeline_execution_schedule_impl(
    const residency::execution::Plan &,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &,
    PipelineExecutionSchedulePrepared &) noexcept;

[[nodiscard]] Status submit_pipeline_execution_schedule_impl(
    const residency::execution::Plan &,
    const PipelineExecutionSchedulePrepared &,
    const residency::ExecutionLease &, PipelineResidencyWindowReleaseCompletion,
    PipelineResidencyScheduleFinalCompletion, void *,
    PipelineResidencyScheduleControl &,
    node::accel::detail::PreparedResidencyStreamControl &) noexcept;

[[nodiscard]] Status
signal_pipeline_execution_schedule_impl(PipelineResidencyScheduleControl &,
                                        std::uint64_t, Status) noexcept;

[[nodiscard]] Status
abort_pipeline_execution_schedule_impl(PipelineResidencyScheduleControl &,
                                       Status) noexcept;

} // namespace rund::compute::detail
