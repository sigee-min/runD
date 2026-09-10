#include "internal.hpp"

#include "../../state.hpp"

#include "../../../cpu/graph.hpp"
#include "../../../job/local.hpp"
#include "../../../status.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstdint>

namespace rund::compute::detail {
namespace {

void accumulate(Stats &total, const Stats &value,
                const std::uint64_t conflict_count,
                const std::uint64_t overflow_ordinal,
                const bool control) noexcept {
  ::rund::detail::counter::Accumulate(total.dispatches, value.dispatches);
  ::rund::detail::counter::Accumulate(total.kernel_ns, value.kernel_ns);
  ::rund::detail::counter::Accumulate(total.kernel_samples,
                                      value.kernel_samples);
  ::rund::detail::counter::Accumulate(total.internal_roundtrip_bytes,
                                      value.internal_roundtrip_bytes);
  ::rund::detail::counter::Accumulate(total.reset_bytes, value.reset_bytes);
  ::rund::detail::counter::Accumulate(total.reset_commands,
                                      value.reset_commands);
  ::rund::detail::counter::Accumulate(total.graph_read_bytes,
                                      value.graph_read_bytes);
  total.worker_count = std::max(total.worker_count, value.worker_count);
  total.participating_workers =
      std::max(total.participating_workers, value.participating_workers);
  ::rund::detail::counter::Accumulate(total.tile_count, value.tile_count);
  total.tile_size = std::max(total.tile_size, value.tile_size);
  ::rund::detail::counter::Accumulate(total.vector_chunks, value.vector_chunks);
  ::rund::detail::counter::Accumulate(total.tail_chunks, value.tail_chunks);
  if (!control) {
    return;
  }
  ::rund::detail::counter::Accumulate(total.control.generated_item_count,
                                      value.control.generated_item_count);
  ::rund::detail::counter::Accumulate(total.control.generated_capacity,
                                      value.control.generated_capacity);
  ::rund::detail::counter::Accumulate(total.control.indirect_dispatch_count,
                                      value.control.indirect_dispatch_count);
  ::rund::detail::counter::Accumulate(total.control.indirect_work_item_count,
                                      value.control.indirect_work_item_count);
  ::rund::detail::counter::Accumulate(total.control.iteration_count,
                                      value.control.iteration_count);
  ::rund::detail::counter::Accumulate(total.control.skipped_iteration_count,
                                      value.control.skipped_iteration_count);
  ::rund::detail::counter::Accumulate(total.control.conflict_count,
                                      conflict_count);
  total.control.overflow_ordinal =
      std::min(total.control.overflow_ordinal, overflow_ordinal);
}

} // namespace

Status consume_cpu_pipeline_step(PipelineState &state, const std::size_t index,
                                 const Status execution) noexcept {
  if (index >= state.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineStep &step = state.steps[index];
  const std::shared_ptr<JobState> &job =
      state.transactional && state.attempt.parity != 0u ? step.alternate_job
                                                        : step.job;
  if (step.program == nullptr || job == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  if (step.program->empty()) {
    ::rund::detail::counter::Accumulate(state.stats.graph_read_bytes,
                                        step.program->graph_info.read_bytes);
    return Status::success();
  }
  if (job->cpu == nullptr || job->cpu->graph == nullptr) {
    return Status::fail(Reason::CpuRunInvalid);
  }
  const CpuGraphRun &graph = *job->cpu->graph;
  const Status semantic =
      graph.semantic_failure_count == 0u || graph.semantic_status == 0u
          ? Status::success()
          : Status::fail(primitive_execution_reason(graph.semantic_primitive,
                                                    graph.semantic_status));
  const Status primary = execution ? semantic : execution;
  const bool resident_overflow =
      step.window != 0u && primary.reason() == Reason::BoundedCountInvalid;
  const Status resolved =
      pipeline_window_status(state.windows, step, primary, state.stats.control);
  accumulate(state.stats, job->cpu->stats, graph.conflict_count,
             graph.overflow_ordinal, !resident_overflow);
  if (resolved && step.window != 0u && step.route == PipelineRoute::Ordinary) {
    const PipelineWindow *const descriptor = state.windows.find(step.window);
    if (descriptor == nullptr ||
        state.windows.progress(step.window - 1u).stopped) {
      return Status::fail(Reason::PipelineInvalid);
    }
    state.windows.progress(step.window - 1u).current =
        (step.iteration & 1u) == 0u ? PipelineWindow::first
                                    : PipelineWindow::second;
    ::rund::detail::counter::Accumulate(state.stats.control.iteration_count,
                                        1u);
  }
  return resolved;
}

} // namespace rund::compute::detail
