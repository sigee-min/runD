#include "local.hpp"

#include "../../../device/state.hpp"

#include <utility>

namespace rund::compute::detail {

Status prepare_cpu_run(JobState &job) {
  if (job.program != nullptr && job.program->empty()) {
    return Status::success();
  }
  if (job.program == nullptr || job.program->cpu_graph == nullptr) {
    job.cpu.reset();
    return Status::success();
  }
  CpuRun &run = job.cpu.emplace();
  const Status materialized =
      materialize_cpu_run(run, job.program, job_graph_buffers(job));
  if (!materialized) {
    job.cpu.reset();
    return materialized;
  }
  return refresh_cpu_map_bindings(job);
}

Status prepare_cpu_run(JobState &job, std::shared_ptr<CpuGraphStorage> storage,
                       const CpuRunRoutePlan &plan,
                       std::shared_ptr<CpuPreparedArena> prepared_arena,
                       const CpuRunRouteSlice &route_slice) {
  if (job.program != nullptr && job.program->empty()) {
    return Status::success();
  }
  if (plan.program == nullptr) {
    job.cpu.reset();
    return Status::success();
  }
  CpuRun &run = job.cpu.emplace();
  const Status materialized = materialize_cpu_run(
      run, job.program, job_graph_buffers(job), std::move(storage), plan,
      std::move(prepared_arena), route_slice);
  if (!materialized) {
    job.cpu.reset();
    return materialized;
  }
  return refresh_cpu_map_bindings(job);
}

} // namespace rund::compute::detail
