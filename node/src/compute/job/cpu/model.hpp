#pragma once

#include <kernel/program/compute/tile/model.hpp>
#include <rund/compute/status.hpp>

#include <atomic>
#include <cstdint>

namespace rund::compute::detail {

struct JobState;
struct RunState;

enum class PassFlow : std::uint8_t {
  Next,
  Repeat,
};

struct PassResult final {
  Status status{Status::success()};
  PassFlow flow{PassFlow::Next};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status);
  }
};

struct StepResult final {
  Status status{Status::success()};
  bool complete{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status);
  }
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
[[nodiscard]] PassResult
finish_graph_pass(JobState &job, const kernel::ComputeTileRunResult *tiles,
                  const std::atomic_bool *cancel) noexcept;

[[nodiscard]] Status initialize_cpu_run(JobState &job) noexcept;
[[nodiscard]] StepResult start_cpu(JobState &job,
                                   const std::atomic_bool *cancel) noexcept;
[[nodiscard]] StepResult finish_cpu(JobState &job,
                                    const kernel::ComputeTileRunResult *tiles,
                                    const std::atomic_bool *cancel) noexcept;
[[nodiscard]] RunState completed_state(JobState &job) noexcept;

} // namespace rund::compute::detail
