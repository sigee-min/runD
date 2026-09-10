#include "local.hpp"

#include "../../../cpu/run/state.hpp"

namespace rund::compute::detail::cpu_tile_detail {
namespace {

void primitive_worker(void *const raw, const kernel::Partition &) noexcept {
  auto *const job = static_cast<JobState *>(raw);
  if (job == nullptr || job->cpu == nullptr) {
    return;
  }
  job->cpu->primitive_status = execute_cpu_primitive(*job, job->cpu->step);
}

void primitive_ready(void *const raw, const bool ok) noexcept {
  auto *const run = static_cast<CpuRun *>(raw);
  if (run == nullptr) {
    return;
  }
  if (!ok) {
    run->primitive_status = Status::fail(Reason::PrimitiveBackendFailed);
  }
  if (run->primitive_ready != nullptr) {
    run->primitive_ready(run->primitive_ready_context);
  }
}

} // namespace

Status submit_primitive(JobState &job, const kernel::WorkerBackend backend,
                        void *const ready_context,
                        void (*const ready)(void *context) noexcept) noexcept {
  if (!backend || backend.submit_partitions == nullptr || job.cpu == nullptr ||
      ready == nullptr) {
    return Status::fail(Reason::PrimitiveBackendInvalid);
  }
  const kernel::u32 width = backend.worker_count == nullptr
                                ? 0u
                                : backend.worker_count(backend.context);
  const kernel::WorkerBackendCapabilities caps =
      kernel::InspectWorkerBackend(backend, width);
  if (!caps.width_matches_request || !caps.supports_async_partitions ||
      caps.is_nested) {
    return Status::fail(Reason::PrimitiveBackendInvalid);
  }
  CpuRun &run = *job.cpu;
  run.primitive_partition =
      kernel::Partition{.worker_index = 0u, .begin = 0u, .end = 1u};
  run.primitive_status = Status::fail(Reason::PrimitiveNotReady);
  run.primitive_ready_context = ready_context;
  run.primitive_ready = ready;
  run.primitive_submission.remaining.store(0u, std::memory_order_relaxed);
  run.primitive_submission.failed.store(false, std::memory_order_relaxed);
  run.primitive_submission.completion = kernel::WorkerCompletion{
      .context = &run,
      .invoke = primitive_ready,
  };
  const bool submitted = backend.submit_partitions(
      backend.context, &run.primitive_partition, 1u,
      kernel::WorkerTask{.context = &job, .invoke = primitive_worker}, nullptr,
      &run.primitive_submission);
  if (!submitted) {
    run.primitive_ready = nullptr;
    run.primitive_ready_context = nullptr;
    return Status::fail(Reason::PrimitiveSubmitFailed);
  }
  return Status::success();
}

} // namespace rund::compute::detail::cpu_tile_detail
