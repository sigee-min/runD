#include "internal.hpp"

#include "../../state.hpp"

#include "../../../../accel/kernel/prepared/interface/api.hpp"

#include <rund/counter.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

bool run_cpu_nested(PipelineState &state, PipelineOutcome &outcome,
                    std::size_t &index) {
  PipelineStep &step = state.steps[index];
  if (step.window == 0u || step.window > state.windows.size()) {
    outcome.status = Status::fail(Reason::PipelineInvalid);
    outcome.failed_step = index;

    return false;
  }
  const PipelineWindow &nested = state.windows[step.window - 1u];
  const node::accel::detail::NestedTemplateShape &shape = nested.nested_shape;
  node::accel::detail::NestedTemplateShape expected_shape{};
  if (!nested.nested() || index != shape.first() ||
      nested.recurrent_output_count == 0u || shape.end() > state.steps.size() ||
      !node::accel::detail::ProveNestedTemplateShape(
          shape.first(), nested.control.maximum, nested.control.tile,
          shape.inner_bound(), expected_shape) ||
      expected_shape != shape) {
    outcome.status = Status::fail(Reason::PipelineInvalid);
    outcome.failed_step = index;

    return false;
  }

  std::size_t executed_outer = 0u;
  for (std::size_t outer = 0u; outer < shape.outer_bound(); ++outer) {
    const PipelineStep &occurrence = state.steps[shape.fold_first()];
    bool active = true;
    const Status ready = prepare_cpu_pipeline_window(
        cpu_pipeline_publication_context(state), state.windows,
        occurrence.window, static_cast<std::uint32_t>(outer),
        state.stats.control, active);
    if (!ready) {
      outcome.status =
          pipeline_window_status(state.windows, occurrence, ready, state.stats.control);
      outcome.failed_step = shape.first();

      state.stats.pipeline.failed_outer_window = outer;
      state.stats.pipeline.failed_nested_phase =
          pipeline_nested_phase(step.route);
      state.windows.progress(step.window - 1u).stopped = true;
      return false;
    }
    if (!active) {
      continue;
    }

    const std::size_t seed_index = shape.seed_first() + outer;
    Status status = execute_cpu_pipeline_step(state, outcome, seed_index);
    if (!status) {
      outcome.status = status;
      outcome.failed_step = seed_index;

      state.stats.pipeline.failed_outer_window = outer;
      state.stats.pipeline.failed_nested_phase =
          pipeline_nested_phase(state.steps[seed_index].route);
      state.windows.progress(step.window - 1u).stopped = true;
      return false;
    }
    for (std::size_t inner = 0u; inner < shape.action_count(); ++inner) {
      const std::size_t action_index = shape.action_first() + inner;
      status = execute_cpu_pipeline_step(state, outcome, action_index);
      if (!status) {
        outcome.status = status;
        outcome.failed_step = action_index;

        state.stats.pipeline.failed_outer_window = outer;
        state.stats.pipeline.failed_inner_iteration = inner;
        state.stats.pipeline.failed_nested_phase =
            pipeline_nested_phase(state.steps[action_index].route);
        state.windows.progress(step.window - 1u).stopped = true;
        return false;
      }
      ++state.stats.pipeline.executed_inner_iteration_count;
    }
    std::uint32_t fold_route = 0u;
    if (!shape.fold_route_for_outer(static_cast<std::uint32_t>(outer),
                                    fold_route)) {
      outcome.status = Status::fail(Reason::PipelineInvalid);
      outcome.failed_step = shape.fold_first();

      return false;
    }
    const std::size_t fold_index = shape.fold_first() + fold_route;
    status = execute_cpu_pipeline_step(state, outcome, fold_index);
    if (!status) {
      outcome.status = status;
      outcome.failed_step = fold_index;

      state.stats.pipeline.failed_outer_window = outer;
      state.stats.pipeline.failed_nested_phase =
          pipeline_nested_phase(state.steps[fold_index].route);
      state.windows.progress(step.window - 1u).stopped = true;
      return false;
    }
    bool window_wrote = false;
    status =
        publish_cpu_pipeline_window(state, step.window, outer, window_wrote);
    outcome.writes_possible = outcome.writes_possible || window_wrote;
    if (!status) {
      outcome.status = status;
      outcome.failed_step = fold_index;

      state.stats.pipeline.failed_outer_window = outer;
      state.stats.pipeline.failed_nested_phase =
          pipeline_nested_phase(state.steps[fold_index].route);
      state.windows.progress(step.window - 1u).stopped = true;
      return false;
    }
    state.windows.progress(step.window - 1u).current =
        fold_route == 1u ? PipelineWindow::second : PipelineWindow::first;
    ++executed_outer;
    ++state.stats.pipeline.executed_outer_window_count;
    ::rund::detail::counter::Accumulate(state.stats.control.iteration_count,
                                        1u);
  }
  const std::size_t skipped = shape.seed_count() - executed_outer;
  ::rund::detail::counter::Accumulate(
      state.stats.pipeline.skipped_outer_window_count,
      static_cast<std::uint64_t>(skipped));
  ::rund::detail::counter::Accumulate(
      state.stats.pipeline.skipped_inner_iteration_count,
      ::rund::detail::counter::SaturatingMultiply(
          static_cast<std::uint64_t>(skipped),
          static_cast<std::uint64_t>(shape.action_count())));
  outcome.verified = shape.end();
  index = shape.end() - 1u;
  return true;
}

} // namespace rund::compute::detail
