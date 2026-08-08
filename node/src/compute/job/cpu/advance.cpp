#include "model.hpp"

#include "../../host.hpp"
#include "../local.hpp"

#include <memory>
#include <utility>

namespace rund::compute::detail {

namespace {

[[nodiscard]] CpuStepProgress
advance_run(const std::shared_ptr<JobState> &state,
            const kernel::WorkerBackend worker_backend,
            const std::atomic_bool *const cancel, void *const ready_context,
            const CpuJobReady ready) noexcept {
  if (state != nullptr && state->program != nullptr &&
      state->program->empty()) {
    return CpuStepProgress::complete();
  }
  if (state == nullptr || state->program == nullptr || state->cpu == nullptr) {
    return CpuStepProgress::failed(Status::fail(Reason::RunInvalid));
  }
  CpuRun &run = *state->cpu;
  if (state->program->cpu_graph == nullptr || run.graph == nullptr) {
    return CpuStepProgress::failed(Status::fail(Reason::RunInvalid));
  }
  kernel::ComputeTileRunResult finished{};
  const kernel::ComputeTileRunResult *view = nullptr;
  if (run.pass != CpuPass::Primitive) {
    kernel::ComputeTileExecutor *const tiles = active_tiles(*state);
    if (tiles == nullptr) {
      return CpuStepProgress::failed(Status::fail(Reason::CpuStepInvalid));
    }
    finished = tiles->finish();
    view = &finished;
  }
  const CpuStepProgress progress = finish_cpu(*state, view, cancel);
  switch (progress.disposition()) {
  case CpuStepDisposition::Failed:
  case CpuStepDisposition::Complete:
    return progress;
  case CpuStepDisposition::Pending:
    break;
  }
  const Status submitted =
      submit_graph_pass(*state, worker_backend, ready_context, ready);
  return submitted ? CpuStepProgress::pending()
                   : CpuStepProgress::failed(submitted);
}

} // namespace

CpuJobProgress advance_cpu_job_on(const std::shared_ptr<JobState> &state,
                                  const kernel::WorkerBackend worker_backend,
                                  const std::atomic_bool *const cancel,
                                  void *const ready_context,
                                  const CpuJobReady ready) noexcept {
  const CpuStepProgress advanced =
      advance_run(state, worker_backend, cancel, ready_context, ready);
  switch (advanced.disposition()) {
  case CpuStepDisposition::Failed:
    return {.status = advanced.status()};
  case CpuStepDisposition::Pending:
    return {};
  case CpuStepDisposition::Complete:
    break;
  }
  auto run =
      state != nullptr && state->program != nullptr && state->program->empty()
          ? empty_job_run(state)
          : Result<RunState>::success(completed_state(*state));
  return run ? CpuJobProgress{.status = Status::success(),
                              .run = std::move(run).value()}
             : CpuJobProgress{.status = Status::fail(run.reason())};
}

CpuStepProgress
advance_cpu_pipeline_job_on(const std::shared_ptr<JobState> &state,
                            const kernel::WorkerBackend worker_backend,
                            const std::atomic_bool *const cancel,
                            void *const ready_context,
                            const CpuJobReady ready) noexcept {
  return advance_run(state, worker_backend, cancel, ready_context, ready);
}

} // namespace rund::compute::detail
