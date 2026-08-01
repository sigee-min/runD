#pragma once

#include "state.hpp"

#include <kernel/dispatch/worker/backend.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {

struct CpuView;

using PipelineCompletion =
    void (*)(void *, node::accel::detail::PreparedPipelineEvidence &&) noexcept;

struct CpuPipelineSchedule final {
  std::size_t step{};
  std::size_t outer{};
};

struct CpuPipelineSelection final {
  Status status{Status::success()};
  std::shared_ptr<JobState> job{};
  std::size_t step{};
  bool complete{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status);
  }
};

[[nodiscard]] bool
valid_pipeline(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] inline std::size_t
logical_verified_steps(const PipelineState &state,
                       const std::size_t physical_prefix) noexcept {
  const std::size_t prefix = physical_prefix < state.steps.size()
                                 ? physical_prefix
                                 : state.steps.size();
  return prefix == state.steps.size()
             ? state.logical_step_count
             : static_cast<std::size_t>(state.steps[prefix].logical_step);
}
[[nodiscard]] inline std::size_t
logical_step_index(const PipelineState &state,
                   const std::size_t physical_index) noexcept {
  return physical_index < state.steps.size()
             ? static_cast<std::size_t>(
                   state.steps[physical_index].logical_step)
             : PipelineStats::no_failed_step;
}
[[nodiscard]] Status
consume_cpu_pipeline_step(PipelineState &state, std::size_t index,
                          Status execution = Status::success()) noexcept;
[[nodiscard]] Status prepare_cpu_pipeline_window(PipelineState &state,
                                                 std::size_t index,
                                                 bool &active) noexcept;
[[nodiscard]] Status prepare_cpu_pipeline_window(PipelineState &state,
                                                 const PipelineStep &step,
                                                 bool &active) noexcept;
[[nodiscard]] Status cpu_resident_view(PipelineState &state,
                                       const PipelineWindow &window,
                                       std::uint32_t output,
                                       CpuView &view) noexcept;
void reset_cpu_resident(PipelineState &state) noexcept;
[[nodiscard]] Status publish_cpu_pipeline(PipelineState &state) noexcept;
[[nodiscard]] Status publish_cpu_pipeline_window(PipelineState &state,
                                                 std::uint16_t window,
                                                 std::size_t outer,
                                                 bool &wrote) noexcept;
[[nodiscard]] std::uint64_t
cpu_program_status_entries(const ProgramState &program) noexcept;
void reset_pipeline_profile(PipelineState &state) noexcept;
void begin_pipeline_profile_step(PipelineState &state,
                                 std::size_t index) noexcept;
void finish_pipeline_profile_step(PipelineState &state,
                                  std::size_t index) noexcept;
void capture_cpu_pipeline_step(PipelineState &state, std::size_t index,
                               bool executed) noexcept;
void capture_accel_pipeline_profile(
    PipelineState &state,
    const node::accel::detail::PreparedPipelineEvidence &evidence,
    bool row_identity_valid) noexcept;

[[nodiscard]] Result<Backend>
pipeline_backend(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] kernel::u32
pipeline_workers(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Stats
pipeline_stats(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Status
queue_pipeline(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Status start_pipeline(PipelineState &state) noexcept;
[[nodiscard]] Status
cancel_pipeline(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Status fail_pipeline(const std::shared_ptr<PipelineState> &state,
                                   Status failure) noexcept;
[[nodiscard]] std::size_t
pipeline_size(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] std::shared_ptr<JobState>
pipeline_job(const std::shared_ptr<PipelineState> &state,
             std::size_t index) noexcept;
[[nodiscard]] Status
begin_pipeline_step(const std::shared_ptr<PipelineState> &state,
                    std::size_t index) noexcept;
[[nodiscard]] Status
complete_pipeline_step(const std::shared_ptr<PipelineState> &state,
                       std::size_t index, Status result) noexcept;
[[nodiscard]] Status
initialize_cpu_pipeline_schedule(const std::shared_ptr<PipelineState> &state,
                                 CpuPipelineSchedule &schedule) noexcept;
[[nodiscard]] CpuPipelineSelection
select_cpu_pipeline_step(const std::shared_ptr<PipelineState> &state,
                         CpuPipelineSchedule &schedule) noexcept;
[[nodiscard]] Status
complete_cpu_pipeline_schedule_step(const std::shared_ptr<PipelineState> &state,
                                    CpuPipelineSchedule &schedule,
                                    Status result) noexcept;
[[nodiscard]] Status
complete_cpu_pipeline(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Status
submit_pipeline_on(const std::shared_ptr<PipelineState> &state,
                   std::shared_ptr<void> lifetime,
                   PipelineCompletion completion, void *user) noexcept;
[[nodiscard]] Status finish_pipeline_on(
    const std::shared_ptr<PipelineState> &state,
    node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept;
[[nodiscard]] Status seed_pipeline_generations(PipelineState &state,
                                               std::uint64_t generation,
                                               std::uint8_t parity) noexcept;
[[nodiscard]] bool rebase_failed_pipeline_generation(PipelineState &state,
                                                     bool submitted,
                                                     Reason failure) noexcept;
void record_pipeline_frame(const std::shared_ptr<PipelineState> &state,
                           std::uint64_t bytes, bool reused,
                           std::uint64_t budget) noexcept;
void release_pipeline_frame(const std::shared_ptr<PipelineState> &state,
                            std::uint64_t bytes) noexcept;

} // namespace rund::compute::detail
