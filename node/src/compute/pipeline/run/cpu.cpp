#include "../local.hpp"
#include "../state.hpp"
#include "cpu/internal.hpp"

namespace rund::compute::detail {

PipelineOutcome run_cpu(PipelineState &state, const std::size_t step_limit) {
  PipelineOutcome outcome{.publication_suppressed = true};

  for (std::size_t index = 0u; index < step_limit; ++index) {
    PipelineStep &step = state.steps[index];
    if (step.route == PipelineRoute::NestedSeed) {
      if (!run_cpu_nested(state, outcome, index)) {
        return outcome;
      }
      continue;
    }
    if (step.route != PipelineRoute::Ordinary) {
      outcome.status = Status::fail(Reason::PipelineInvalid);
      outcome.failed_step = index;

      return outcome;
    }
    if (!run_cpu_ordinary_step(state, outcome, index)) {
      return outcome;
    }
  }

  const Status published = publish_cpu_pipeline(state);
  if (!published) {
    outcome.status = published;
    outcome.failed_step = state.steps.empty() ? 0u : state.steps.size() - 1u;
    if (state.steps.empty()) {
      outcome.failed_step.reset();
    }
    outcome.writes_possible = true;
    outcome.publication_suppressed = false;
    return outcome;
  }
  state.stats.command_submits = 0u;
  state.stats.pipeline.verified_step_count =
      logical_verified_steps(state, outcome.verified);
  state.stats.pipeline.failed_step_index = PipelineStats::no_failed_step;
  return outcome;
}

} // namespace rund::compute::detail
