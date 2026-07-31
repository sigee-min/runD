#include "model.hpp"

#include "../../host.hpp"
#include "../local.hpp"

#include <memory>

namespace rund::compute::detail {

namespace {

[[nodiscard]] Status validate_submission(const std::shared_ptr<JobState> &state,
                                         const kernel::u32 workers) noexcept {
  const Result<Backend> backend = job_backend(state);
  if (!backend) {
    return Status::fail(backend.reason());
  }
  if (*backend != Backend::Cpu) {
    return Status::fail(Reason::NodeHostAccelNotPrepared);
  }
  if (job_workers(state) != workers) {
    return Status::fail(Reason::NodeHostWidthMismatch);
  }
  return Status::success();
}

[[nodiscard]] Status submit_run(const std::shared_ptr<JobState> &state,
                                const kernel::WorkerBackend worker_backend,
                                const std::atomic_bool *const cancel,
                                void *const ready_context,
                                const CpuJobReady ready) noexcept {
  if (state->program->empty()) {
    ready(ready_context);
    return Status::success();
  }
  // Public submission crossed queue_job and private Pipeline submission
  // crossed valid_job in its entry point. This transport owns only O(1)
  // prepared-state checks and the common CPU transition owner.
  if (state->cpu == nullptr || state->program->cpu_graph == nullptr ||
      state->cpu->graph == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  const StepResult started = start_cpu(*state, cancel);
  if (!started) {
    return started.status;
  }
  if (started.complete) {
    ready(ready_context);
    return Status::success();
  }
  return submit_graph_pass(*state, worker_backend, ready_context, ready);
}

} // namespace

Status submit_cpu_job_on(const std::shared_ptr<JobState> &state,
                         const kernel::WorkerBackend worker_backend,
                         const kernel::u32 workers,
                         const std::atomic_bool *const cancel,
                         void *const ready_context,
                         const CpuJobReady ready) noexcept {
  const Status valid = validate_submission(state, workers);
  if (!valid) {
    return valid;
  }
  const Status started = start_queued_job(state);
  if (!started) {
    return started;
  }
  return submit_run(state, worker_backend, cancel, ready_context, ready);
}

Status submit_cpu_pipeline_job_on(const std::shared_ptr<JobState> &state,
                                  const kernel::WorkerBackend worker_backend,
                                  const kernel::u32 workers,
                                  const std::atomic_bool *const cancel,
                                  void *const ready_context,
                                  const CpuJobReady ready) noexcept {
  const Status valid = validate_submission(state, workers);
  if (!valid) {
    return valid;
  }
  if (!valid_job(state) || state->terminal != nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const Status gathered = gather_cpu_pipeline_views(state);
  if (!gathered) {
    return gathered;
  }
  return submit_run(state, worker_backend, cancel, ready_context, ready);
}

} // namespace rund::compute::detail
