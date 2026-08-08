#pragma once

#include "job/state.hpp"

#include <kernel/dispatch/worker/backend.hpp>

#include <atomic>
#include <memory>
#include <span>

namespace rund::compute::detail {

class CpuStepProgress;

[[nodiscard]] Result<RunState>
run_buffers(const std::shared_ptr<ProgramState> &program,
            std::span<const std::shared_ptr<BufferState>> inputs,
            std::span<const std::shared_ptr<BufferState>> outputs);

using CpuJobReady = void (*)(void *context) noexcept;

struct CpuJobProgress final {
  Status status{Status::success()};
  std::optional<RunState> run{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status);
  }

  [[nodiscard]] bool complete() const noexcept { return run.has_value(); }
};

[[nodiscard]] Status submit_cpu_job_on(const std::shared_ptr<JobState> &state,
                                       kernel::WorkerBackend worker_backend,
                                       kernel::u32 workers,
                                       const std::atomic_bool *cancel,
                                       void *ready_context,
                                       CpuJobReady ready) noexcept;
// Pipeline-private Jobs are already exclusively owned by the Pipeline phase
// gate and therefore start without reacquiring or mutating a public Job phase.
[[nodiscard]] Status
submit_cpu_pipeline_job_on(const std::shared_ptr<JobState> &state,
                           kernel::WorkerBackend worker_backend,
                           kernel::u32 workers, const std::atomic_bool *cancel,
                           void *ready_context, CpuJobReady ready) noexcept;

[[nodiscard]] CpuJobProgress
advance_cpu_job_on(const std::shared_ptr<JobState> &state,
                   kernel::WorkerBackend worker_backend,
                   const std::atomic_bool *cancel, void *ready_context,
                   CpuJobReady ready) noexcept;
[[nodiscard]] CpuStepProgress
advance_cpu_pipeline_job_on(const std::shared_ptr<JobState> &state,
                            kernel::WorkerBackend worker_backend,
                            const std::atomic_bool *cancel, void *ready_context,
                            CpuJobReady ready) noexcept;

[[nodiscard]] Status submit_job_on(const std::shared_ptr<JobState> &state,
                                   std::shared_ptr<void> lifetime,
                                   JobCompletion completion,
                                   void *user) noexcept;

[[nodiscard]] Result<Backend>
job_backend(const std::shared_ptr<JobState> &state) noexcept;

[[nodiscard]] kernel::u32
job_workers(const std::shared_ptr<JobState> &state) noexcept;

} // namespace rund::compute::detail
