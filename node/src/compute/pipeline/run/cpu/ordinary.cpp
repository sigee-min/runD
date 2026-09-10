#include "internal.hpp"

#include "../../state.hpp"

#include <cstddef>

namespace rund::compute::detail {

Status run_cpu_ordinary_step(PipelineState &state, PipelineOutcome &outcome,
                             const std::size_t index) {
  PipelineStep &step = state.steps[index];
  bool active = true;
  const Status window_ready = prepare_cpu_pipeline_window(state, index, active);
  if (!window_ready) {
    outcome.status = pipeline_window_status(state.windows, step, window_ready,
                                            state.stats.control);
    outcome.failed_step = index;

    return outcome.status;
  }
  if (!active) {
    ++outcome.verified;
    return Status::success();
  }
  const Status executed = execute_cpu_pipeline_step(state, outcome, index);
  if (!executed) {
    outcome.status = pipeline_window_status(state.windows, step, executed,
                                            state.stats.control);
    outcome.failed_step = index;

    return outcome.status;
  }
  ++outcome.verified;
  return Status::success();
}

} // namespace rund::compute::detail
