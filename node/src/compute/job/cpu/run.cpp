#include "model.hpp"

#include "../../cpu/run/state.hpp"

#include <algorithm>
#include <utility>

namespace rund::compute::detail {

RunState completed_state(JobState &job) noexcept {
  RunState result{};
  result.program = job.program;
  std::copy(job.outputs.begin(), job.outputs.end(), result.outputs.begin());
  result.stats = job.cpu->stats;
  result.stats.control.conflict_count = job.cpu->graph->conflict_count;
  result.stats.control.overflow_ordinal = job.cpu->graph->overflow_ordinal;
  result.semantic_primitive = job.cpu->graph->semantic_primitive;
  result.semantic_status = job.cpu->graph->semantic_status;
  result.semantic_failure_count = job.cpu->graph->semantic_failure_count;
  return result;
}

Status run_cpu_pipeline_job(JobState &job) {
  StepResult progress = start_cpu(job, nullptr);
  while (progress && !progress.complete) {
    if (job.cpu->pass == CpuPass::Primitive) {
      job.cpu->primitive_status = execute_cpu_primitive(job, job.cpu->step);
      progress = finish_cpu(job, nullptr, nullptr);
    } else {
      const kernel::ComputeTileRunResult tiles = run_graph_pass(job);
      progress = finish_cpu(job, &tiles, nullptr);
    }
  }
  return progress.status;
}

Result<RunState> run_cpu_job(JobState &job) {
  const Status ran = run_cpu_pipeline_job(job);
  return ran ? Result<RunState>::success(completed_state(job))
             : Result<RunState>::fail(ran.reason());
}

} // namespace rund::compute::detail
