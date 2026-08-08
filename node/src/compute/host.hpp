#pragma once

#include "job/state.hpp"

#include <kernel/dispatch/worker/backend.hpp>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::compute::detail {

class CpuStepProgress;

[[nodiscard]] Result<RunState>
run_buffers(const std::shared_ptr<ProgramState> &program,
            std::span<const std::shared_ptr<BufferState>> inputs,
            std::span<const std::shared_ptr<BufferState>> outputs);

using CpuJobReady = void (*)(void *context) noexcept;

enum class CpuJobProgressDisposition : std::uint8_t {
  Failed,
  Pending,
  Complete,
};

class CpuJobProgress final {
public:
  CpuJobProgress() = delete;
  CpuJobProgress(const CpuJobProgress &) = delete;
  CpuJobProgress &operator=(const CpuJobProgress &) = delete;

  CpuJobProgress(CpuJobProgress &&other) noexcept
      : disposition_(std::exchange(other.disposition_,
                                   CpuJobProgressDisposition::Failed)),
        failure_(
            std::exchange(other.failure_, Status::fail(Reason::RunInvalid))),
        run_(std::move(other.run_)) {
    other.run_.reset();
  }

  CpuJobProgress &operator=(CpuJobProgress &&) = delete;

  [[nodiscard]] static CpuJobProgress failed(Status failure) noexcept {
    if (failure) {
      failure = Status::fail(Reason::RunInvalid);
    }
    return CpuJobProgress{failure};
  }

  [[nodiscard]] static CpuJobProgress pending() noexcept {
    return CpuJobProgress{PendingTag{}};
  }

  [[nodiscard]] static CpuJobProgress complete(RunState &&run) noexcept {
    return CpuJobProgress{std::move(run)};
  }

  [[nodiscard]] CpuJobProgressDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] Status status() const noexcept {
    return disposition_ == CpuJobProgressDisposition::Failed
               ? failure_
               : Status::success();
  }

  [[nodiscard]] RunState take_run() && noexcept {
    if (disposition_ != CpuJobProgressDisposition::Complete ||
        !run_.has_value()) {
      std::abort();
    }
    RunState result = std::move(*run_);
    run_.reset();
    disposition_ = CpuJobProgressDisposition::Failed;
    failure_ = Status::fail(Reason::RunInvalid);
    return result;
  }

private:
  struct PendingTag final {};

  explicit CpuJobProgress(const Status failure) noexcept : failure_(failure) {}

  explicit CpuJobProgress(const PendingTag) noexcept
      : disposition_(CpuJobProgressDisposition::Pending) {}

  explicit CpuJobProgress(RunState &&run) noexcept
      : disposition_(CpuJobProgressDisposition::Complete),
        run_(std::in_place, std::move(run)) {}

  CpuJobProgressDisposition disposition_ = CpuJobProgressDisposition::Failed;
  Status failure_ = Status::fail(Reason::RunInvalid);
  std::optional<RunState> run_{};
};

static_assert(std::is_nothrow_move_constructible_v<RunState>);
static_assert(std::is_nothrow_move_assignable_v<RunState>);

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
