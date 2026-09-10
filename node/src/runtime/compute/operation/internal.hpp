#pragma once

#include "../local.hpp"

namespace rund::node::compute_operation_detail {

void CompleteAsync(void *raw,
                   compute::Result<compute::detail::RunState> result) noexcept;
void CompletePipeline(
    void *raw,
    node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept;
void CpuReady(void *raw) noexcept;

[[nodiscard]] node::accel::detail::KernelTiming
KernelTimingFor(const compute_detail::TaskState &task) noexcept;
[[nodiscard]] bool
CaptureTerminalProfile(const compute_detail::TaskState &task) noexcept;

[[nodiscard]] compute::Result<compute::Backend>
JobBackend(const std::shared_ptr<void> &owner) noexcept;
[[nodiscard]] kernel::u32
JobWorkers(const std::shared_ptr<void> &owner) noexcept;
[[nodiscard]] compute::Status
ReserveJob(const std::shared_ptr<void> &owner) noexcept;
[[nodiscard]] compute_detail::Dispatch
SubmitJobCpu(const compute_detail::Operation &operation,
             compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute_detail::Advance
AdvanceJobCpu(const compute_detail::Operation &operation,
              compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
ResultJobCpu(const compute_detail::Operation &operation,
             compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute_detail::Dispatch
SubmitJobAccel(const compute_detail::Operation &operation,
               compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
ResultJobAccel(const compute_detail::Operation &operation,
               compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
FailJob(const compute_detail::Operation &operation,
        compute_detail::TaskState &task, compute::Status failure) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
CancelJob(const compute_detail::Operation &operation,
          compute_detail::TaskState &task) noexcept;
void RecordJobFrame(const std::shared_ptr<void> &owner, std::uint64_t bytes,
                    bool reused, std::uint64_t budget) noexcept;

[[nodiscard]] compute::Result<compute::Backend>
PipelineBackend(const std::shared_ptr<void> &owner) noexcept;
[[nodiscard]] kernel::u32
PipelineWorkers(const std::shared_ptr<void> &owner) noexcept;
[[nodiscard]] compute::Status
ReservePipeline(const std::shared_ptr<void> &owner) noexcept;
[[nodiscard]] compute_detail::Dispatch
SubmitPipelineCpu(const compute_detail::Operation &operation,
                  compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute_detail::Advance
AdvancePipelineCpu(const compute_detail::Operation &operation,
                   compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
ResultPipelineCpu(const compute_detail::Operation &operation,
                  compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute_detail::Dispatch
SubmitPipelineAccel(const compute_detail::Operation &operation,
                    compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
ResultPipelineAccel(const compute_detail::Operation &operation,
                    compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
FailPipeline(const compute_detail::Operation &operation,
             compute_detail::TaskState &task, compute::Status failure) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
CancelPipeline(const compute_detail::Operation &operation,
               compute_detail::TaskState &task) noexcept;
void RecordPipelineFrame(const std::shared_ptr<void> &owner,
                         std::uint64_t bytes, bool reused,
                         std::uint64_t budget) noexcept;

[[nodiscard]] compute::Result<compute::Backend>
VirtualBackend(const std::shared_ptr<void> &owner) noexcept;
[[nodiscard]] kernel::u32
VirtualWorkers(const std::shared_ptr<void> &owner) noexcept;
[[nodiscard]] compute::Status
ReserveVirtual(const std::shared_ptr<void> &owner) noexcept;
[[nodiscard]] compute_detail::Dispatch
SubmitVirtual(const compute_detail::Operation &operation,
              compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute_detail::Advance
AdvanceVirtual(const compute_detail::Operation &operation,
               compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
ResultVirtual(const compute_detail::Operation &operation,
              compute_detail::TaskState &task) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
FailVirtual(const compute_detail::Operation &operation,
            compute_detail::TaskState &task, compute::Status failure) noexcept;
[[nodiscard]] compute::detail::TerminalObservation
CancelVirtual(const compute_detail::Operation &operation,
              compute_detail::TaskState &task) noexcept;

} // namespace rund::node::compute_operation_detail
