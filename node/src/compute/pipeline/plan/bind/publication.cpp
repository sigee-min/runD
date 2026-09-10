#include "../local.hpp"

#include "../../../job/local.hpp"
#include "../../claim.hpp"
#include "../../state.hpp"
#include "../compare.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {
namespace {


[[nodiscard]] bool
job_binding_matches(const PipelineState &state,
                    const PipelinePublicationViewPlan &planned,
                    const std::shared_ptr<BufferState> &owner,
                    const JobBufferView &view, const bool alternate) noexcept {
  const PipelinePublicationViewIdentity &identity = planned.identity;
  const PipelineResource *const resource =
      selected_pipeline_resource(state, identity.resource_ordinal, alternate);
  return resource != nullptr && resource->buffer != nullptr &&
         owner == resource->buffer && resource->type == planned.type &&
         resource->format == planned.format &&
         resource->bytes == identity.backing_bytes &&
         view.count == identity.count &&
         view.element_bytes == identity.element_bytes &&
         view.offset == planned.offset && view.stride == planned.stride;
}

[[nodiscard]] bool job_step_binding_matches(
    const PipelineState &state, const std::size_t step_index, const bool input,
    const std::uint32_t slot, const PipelinePublicationViewPlan &planned,
    const bool alternate) noexcept {
  if (step_index >= state.steps.size()) {
    return false;
  }
  const std::shared_ptr<JobState> &job =
      alternate ? state.steps[step_index].alternate_job
                : state.steps[step_index].job;
  if (job == nullptr) {
    return false;
  }
  const auto &owners = input ? job->inputs : job->outputs;
  const auto &views = input ? job->input_views : job->output_views;
  return owners.size() == views.size() && slot < owners.size() &&
         job_binding_matches(state, planned, owners[slot], views[slot],
                             alternate);
}

[[nodiscard]] bool job_count_matches(const PipelineState &state,
                                     const std::size_t step_index,
                                     const PipelinePublicationViewPlan &count,
                                     const std::uint32_t count_input,
                                     const bool alternate) noexcept {
  if (step_index >= state.steps.size()) {
    return false;
  }
  const std::shared_ptr<JobState> &job =
      alternate ? state.steps[step_index].alternate_job
                : state.steps[step_index].job;
  return job != nullptr && count_input < job->inputs.size() &&
         job_step_binding_matches(state, step_index, true, count_input, count,
                                  alternate);
}

[[nodiscard]] bool
terminal_step_bindings_match(const PipelineState &state,
                             const PipelineWindow &window,
                             const PipelineTerminalPublicationPlan &terminal,
                             const bool alternate) noexcept {
  if (terminal.output.value >= window.recurrent_output_count) {
    return false;
  }
  if (window.nested()) {
    const std::size_t fold_first = window.nested_shape.fold_first();
    constexpr std::array input_banks{
        PipelineWindow::seed, PipelineWindow::first, PipelineWindow::second};
    constexpr std::array output_banks{
        PipelineWindow::first, PipelineWindow::second, PipelineWindow::first};
    const std::size_t fold_count = window.nested_shape.fold_count();
    if (!window.nested_shape.valid() || fold_count != input_banks.size() ||
        fold_count != output_banks.size() || fold_first > state.steps.size() ||
        state.steps.size() - fold_first < fold_count) {
      return false;
    }
    for (std::size_t route = 0u; route < fold_count; ++route) {
      const std::size_t step_index = fold_first + route;
      if (!job_step_binding_matches(
              state, step_index, true, terminal.output.value,
              terminal.sources[input_banks[route]], alternate) ||
          !job_step_binding_matches(
              state, step_index, false, terminal.output.value,
              terminal.sources[output_banks[route]], alternate)) {
        return false;
      }
    }
    return true;
  }

  bool saw_step = false;
  const std::uint16_t window_index =
      static_cast<std::uint16_t>(terminal.state + 1u);
  for (std::size_t step_index = 0u; step_index < state.steps.size();
       ++step_index) {
    const PipelineStep &step = state.steps[step_index];
    if (step.window != window_index || step.route != PipelineRoute::Ordinary) {
      continue;
    }
    saw_step = true;
    const std::uint32_t input_bank =
        step.iteration == 0u
            ? PipelineWindow::seed
            : ((step.iteration & 1u) != 0u ? PipelineWindow::first
                                           : PipelineWindow::second);
    const std::uint32_t output_bank = (step.iteration & 1u) == 0u
                                          ? PipelineWindow::first
                                          : PipelineWindow::second;
    if (!job_step_binding_matches(state, step_index, true,
                                  terminal.output.value,
                                  terminal.sources[input_bank], alternate) ||
        !job_step_binding_matches(state, step_index, false,
                                  terminal.output.value,
                                  terminal.sources[output_bank], alternate)) {
      return false;
    }
  }
  return saw_step;
}

} // namespace

[[nodiscard]] Status
validate_publication_job_bindings(const PipelineState &state) noexcept {
  for (std::size_t state_index = 0u; state_index < state.windows.size();
       ++state_index) {
    const PipelineWindow &window = state.windows[state_index];
    const std::uint16_t window_index =
        static_cast<std::uint16_t>(state_index + 1u);
    const std::size_t count_step =
        window.nested() ? window.nested_shape.seed_first() : window.first_step;
    if (count_step >= state.steps.size() ||
        state.steps[count_step].window != window_index) {
      return Status::fail(Reason::PipelineInvalid);
    }
    for (const bool alternate : {false, true}) {
      if (alternate && !state.transactional) {
        continue;
      }
      bool saw_count = false;
      for (std::size_t step_index = 0u; step_index < state.steps.size();
           ++step_index) {
        const PipelineStep &step = state.steps[step_index];
        const bool consumes_count =
            step.window == window_index &&
            (!window.nested() || step.route == PipelineRoute::NestedSeed);
        if (!consumes_count) {
          continue;
        }
        saw_count = true;
        if (!job_count_matches(state, step_index, window.control.count,
                               window.control.count_input, alternate)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
      if (!saw_count) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
  }

  std::array<std::uint32_t, PipelineStepCapacity> terminal_counts{};
  if (state.windows.size() > terminal_counts.size()) {
    return Status::fail(Reason::PipelineCapacity);
  }
  for (const PipelinePublicationPlan &publication : state.publications) {
    if (const auto *terminal =
            std::get_if<PipelineTerminalPublicationPlan>(&publication)) {
      if (terminal->state >= state.windows.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      ++terminal_counts[terminal->state];
      const PipelineWindow &window = state.windows[terminal->state];
      for (const bool alternate : {false, true}) {
        if (alternate && !state.transactional) {
          continue;
        }
        if (!terminal_step_bindings_match(state, window, *terminal,
                                          alternate)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
      continue;
    }
    const auto &window_publication =
        std::get<PipelineWindowPublicationPlan>(publication);
    if (window_publication.state >= state.windows.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineWindow &window = state.windows[window_publication.state];
    for (const bool alternate : {false, true}) {
      if (alternate && !state.transactional) {
        continue;
      }
      const std::size_t fold_first = window.nested_shape.fold_first();
      const std::size_t fold_count = window.nested_shape.fold_count();
      if (!window.nested() || !window.nested_shape.valid() ||
          fold_first > state.steps.size() ||
          state.steps.size() - fold_first < fold_count) {
        return Status::fail(Reason::PipelineInvalid);
      }
      for (std::size_t route = 0u; route < fold_count; ++route) {
        if (!job_step_binding_matches(state, fold_first + route, false,
                                      window_publication.output.value,
                                      window_publication.source, alternate)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
    }
  }
  for (std::size_t state_index = 0u; state_index < state.windows.size();
       ++state_index) {
    if (terminal_counts[state_index] !=
        state.windows[state_index].recurrent_output_count) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  return Status::success();
}


} // namespace rund::compute::detail
