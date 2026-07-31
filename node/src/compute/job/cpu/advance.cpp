#include "model.hpp"

#include "../../host.hpp"
#include "../local.hpp"

#include <memory>
#include <utility>

namespace rund::compute::detail {

namespace {

[[nodiscard]] StepResult advance_run(const std::shared_ptr<JobState> &state,
                                     const kernel::WorkerBackend worker_backend,
                                     const std::atomic_bool *const cancel,
                                     void *const ready_context,
                                     const CpuJobReady ready) noexcept {
  if (state != nullptr && state->program != nullptr &&
      state->program->empty()) {
    return {.complete = true};
  }
  if (state == nullptr || state->program == nullptr || state->cpu == nullptr) {
    return {.status = Status::fail(Reason::RunInvalid)};
  }
  CpuRun &run = *state->cpu;
  if (state->program->cpu_graph == nullptr || run.graph == nullptr) {
    return {.status = Status::fail(Reason::RunInvalid)};
  }
  kernel::ComputeTileRunResult finished{};
  const kernel::ComputeTileRunResult *view = nullptr;
  if (run.pass != CpuPass::Primitive) {
    kernel::ComputeTileExecutor *const tiles = active_tiles(*state);
    if (tiles == nullptr) {
      return {.status = Status::fail(Reason::CpuStepInvalid)};
    }
    finished = tiles->finish();
    view = &finished;
  }
  const StepResult progress = finish_cpu(*state, view, cancel);
  if (!progress || progress.complete) {
    return progress;
  }
  return {.status =
              submit_graph_pass(*state, worker_backend, ready_context, ready)};
}

} // namespace

CpuJobProgress advance_cpu_job_on(const std::shared_ptr<JobState> &state,
                                  const kernel::WorkerBackend worker_backend,
                                  const std::atomic_bool *const cancel,
                                  void *const ready_context,
                                  const CpuJobReady ready) noexcept {
  const StepResult advanced =
      advance_run(state, worker_backend, cancel, ready_context, ready);
  if (!advanced || !advanced.complete) {
    return {.status = advanced.status};
  }
  auto run =
      state != nullptr && state->program != nullptr && state->program->empty()
          ? empty_job_run(state)
          : Result<RunState>::success(completed_state(*state));
  return run ? CpuJobProgress{.status = Status::success(),
                              .run = std::move(run).value()}
             : CpuJobProgress{.status = Status::fail(run.reason())};
}

CpuPipelineProgress
advance_cpu_pipeline_job_on(const std::shared_ptr<JobState> &state,
                            const kernel::WorkerBackend worker_backend,
                            const std::atomic_bool *const cancel,
                            void *const ready_context,
                            const CpuJobReady ready) noexcept {
  const StepResult advanced =
      advance_run(state, worker_backend, cancel, ready_context, ready);
  if (!advanced || !advanced.complete) {
    return {.status = advanced.status};
  }
  return {.completed = true};
}

} // namespace rund::compute::detail
