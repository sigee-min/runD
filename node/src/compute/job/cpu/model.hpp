#pragma once

#include <kernel/program/compute/tile/model.hpp>
#include <rund/compute/status.hpp>

#include <atomic>
#include <cstdint>

namespace rund::compute::detail {

struct JobState;
struct RunState;

enum class CpuPassDisposition : std::uint8_t {
  Failed,
  Next,
  Repeat,
};

class CpuPassResult final {
public:
  [[nodiscard]] static constexpr CpuPassResult failed(Status failure) noexcept {
    if (failure) {
      failure = Status::fail(Reason::CpuStepInvalid);
    }
    return CpuPassResult{failure};
  }

  [[nodiscard]] static constexpr CpuPassResult next() noexcept {
    return CpuPassResult{CpuPassDisposition::Next};
  }

  [[nodiscard]] static constexpr CpuPassResult repeat() noexcept {
    return CpuPassResult{CpuPassDisposition::Repeat};
  }

  [[nodiscard]] constexpr CpuPassDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr Status status() const noexcept {
    return disposition_ == CpuPassDisposition::Failed ? failure_
                                                      : Status::success();
  }

private:
  explicit constexpr CpuPassResult(const Status failure) noexcept
      : disposition_{CpuPassDisposition::Failed}, failure_{failure} {}

  explicit constexpr CpuPassResult(
      const CpuPassDisposition disposition) noexcept
      : disposition_{disposition} {}

  CpuPassDisposition disposition_ = CpuPassDisposition::Failed;
  Status failure_ = Status::fail(Reason::CpuStepInvalid);
};

enum class CpuStepDisposition : std::uint8_t {
  Failed,
  Pending,
  Complete,
};

class CpuStepProgress final {
public:
  [[nodiscard]] static constexpr CpuStepProgress
  failed(Status failure) noexcept {
    if (failure) {
      failure = Status::fail(Reason::CpuStepInvalid);
    }
    return CpuStepProgress{failure};
  }

  [[nodiscard]] static constexpr CpuStepProgress pending() noexcept {
    return CpuStepProgress{CpuStepDisposition::Pending};
  }

  [[nodiscard]] static constexpr CpuStepProgress complete() noexcept {
    return CpuStepProgress{CpuStepDisposition::Complete};
  }

  [[nodiscard]] constexpr CpuStepDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr Status status() const noexcept {
    return disposition_ == CpuStepDisposition::Failed ? failure_
                                                      : Status::success();
  }

private:
  explicit constexpr CpuStepProgress(const Status failure) noexcept
      : disposition_{CpuStepDisposition::Failed}, failure_{failure} {}

  explicit constexpr CpuStepProgress(
      const CpuStepDisposition disposition) noexcept
      : disposition_{disposition} {}

  CpuStepDisposition disposition_ = CpuStepDisposition::Failed;
  Status failure_ = Status::fail(Reason::CpuStepInvalid);
};

[[nodiscard]] Status
prepare_graph_step(JobState &job, const std::atomic_bool *cancel) noexcept;
[[nodiscard]] kernel::ComputeTileExecutor *active_tiles(JobState &job) noexcept;
[[nodiscard]] std::uint64_t active_tile_size(const JobState &job) noexcept;
[[nodiscard]] Status
submit_graph_pass(JobState &job, kernel::WorkerBackend backend,
                  void *ready_context,
                  void (*ready)(void *context) noexcept) noexcept;
[[nodiscard]] kernel::ComputeTileRunResult run_graph_pass(JobState &job);
[[nodiscard]] CpuPassResult
finish_graph_pass(JobState &job, const kernel::ComputeTileRunResult *tiles,
                  const std::atomic_bool *cancel) noexcept;

[[nodiscard]] Status initialize_cpu_run(JobState &job) noexcept;
[[nodiscard]] CpuStepProgress
start_cpu(JobState &job, const std::atomic_bool *cancel) noexcept;
[[nodiscard]] CpuStepProgress
finish_cpu(JobState &job, const kernel::ComputeTileRunResult *tiles,
           const std::atomic_bool *cancel) noexcept;
[[nodiscard]] RunState completed_state(JobState &job) noexcept;

} // namespace rund::compute::detail
