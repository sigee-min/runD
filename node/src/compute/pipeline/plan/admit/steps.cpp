#include "../../state/assembly.hpp"
#include "model.hpp"
#include "../resource.hpp"

#include "../../local.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

namespace rund::compute::detail {

using admit::admit_resolved_view;
using admit::physical_output_view;
using admit::publication_matches_resolved;
using admit::validate_step_aliases;

Status admit_steps(AdmissionDraft &draft) {
  PipelineState *const state = draft.state.get();
  const PipelineMemoryPlan &plan = draft.plan;
  const PipelineBuildSnapshot &frozen = draft.frozen;
  auto &hash = draft.hash;
  auto &initialized_windows = draft.initialized_windows;
  std::size_t &observed_bindings = draft.observed_bindings;
  std::uint64_t &status_entry_count = draft.status_entry_count;
  std::size_t &output_count = draft.output_count;
  const std::size_t binding_capacity = draft.binding_capacity;

  for (std::size_t step_index = 0u; step_index < frozen.steps.size();
       ++step_index) {
    const PipelineFrozenStep &declared = frozen.steps[step_index];
    const PipelineStepResourcePlan &sealed = plan.step_resources[step_index];
    const ProgramState *const program = declared.program.get();
    if (program == nullptr || program->device != state->device) {
      return Status::fail(program == nullptr ? Reason::ProgramInvalid
                                             : Reason::BindingDeviceMismatch);
    }
    status_entry_count = ::rund::detail::counter::SaturatingAdd(
        status_entry_count, cpu_program_status_entries(*program));
    if (!valid_input_shape(*program) ||
        sealed.inputs.size() != program->input_types.size()) {
      return Status::fail(Reason::BindingCountMismatch);
    }
    if (sealed.outputs.size() > PipelineLeafCapacity ||
        sealed.inputs.size() > PipelineLeafCapacity ||
        sealed.outputs.size() > PipelineLeafCapacity - sealed.inputs.size() ||
        sealed.physical_sources.size() != program->output_types.size() ||
        program->output_sizes.size() != sealed.physical_sources.size() ||
        program->output_formats.size() != sealed.physical_sources.size()) {
      return Status::fail(Reason::PipelineCapacity);
    }
    if (observed_bindings > binding_capacity ||
        sealed.inputs.size() + sealed.outputs.size() >
            binding_capacity - observed_bindings) {
      return Status::fail(Reason::PipelineCapacity);
    }
    observed_bindings += sealed.inputs.size() + sealed.outputs.size();

    PipelineStep &step = state->steps[step_index];
    step.program = declared.program;
    step.logical_step = declared.logical_step;
    step.iteration = declared.iteration;
    step.iteration_bound = declared.iteration_bound;
    step.route = declared.route;
    step.writes_each_iteration = declared.writes_each_iteration;

    if (declared.nested != 0u) {
      const std::size_t nested_index = declared.nested - 1u;
      if (nested_index >= frozen.nested_windows.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineFrozenNestedWindow &nested =
          frozen.nested_windows[nested_index];
      const node::accel::detail::NestedTemplateShape &shape = nested.shape;
      const std::uint32_t control_index = plan.window_states[step_index];
      if (control_index >= plan.window_controls.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineWindowControl &control =
          plan.window_controls[control_index];
      node::accel::detail::NestedTemplateShape expected_shape{};
      node::accel::detail::NestedTemplateRouteProjection route{};
      if (!shape.valid() || shape.end() > frozen.steps.size() ||
          !shape.project(step_index, route) ||
          !node::accel::detail::ProveNestedTemplateShape(
              shape.first(), control.maximum, control.tile, shape.inner_bound(),
              expected_shape) ||
          expected_shape != shape ||
          declared.route != pipeline_route(route.phase) ||
          declared.iteration != route.iteration ||
          declared.iteration_bound != route.bound ||
          nested.recurrent_output_count == 0u ||
          nested.recurrent_output_count > PipelineLeafCapacity ||
          shape.seed_first() >= plan.step_resources.size() ||
          control.count_input >=
              plan.step_resources[shape.seed_first()].inputs.size() ||
          !publication_matches_resolved(control.count,
                                        plan.step_resources[shape.seed_first()]
                                            .inputs[control.count_input]) ||
          control.count.type != Type::U32 ||
          control.count.identity.count != 1u ||
          control.count.identity.element_bytes != sizeof(std::uint32_t) ||
          control.final < PipelineWindow::first ||
          control.final > PipelineWindow::second) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (step_index == shape.first()) {
        if (initialized_windows[control_index] ||
            shape.fold_first() >= plan.step_resources.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        const PipelineStepResourcePlan &fold =
            plan.step_resources[shape.fold_first()];
        if (nested.recurrent_output_count > fold.physical_sources.size() ||
            (control.terminal_output !=
                 std::numeric_limits<std::uint32_t>::max() &&
             control.terminal_output >= fold.physical_sources.size())) {
          return Status::fail(Reason::PipelineInvalid);
        }
        for (std::size_t output = 0u; output < nested.recurrent_output_count;
             ++output) {
          if (output >= fold.outputs.size() ||
              fold.outputs[output].physical != output) {
            return Status::fail(Reason::PipelineInvalid);
          }
        }
        state->windows.entries_[control_index].descriptor = PipelineWindow{
            .control = control,
            .first_step = shape.fold_first(),
            .recurrent_output_count =
                static_cast<std::uint32_t>(nested.recurrent_output_count),
            .nested_shape = shape,
        };
        initialized_windows[control_index] = true;
      }
      if (!initialized_windows[control_index]) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (declared.route == PipelineRoute::NestedSeed &&
          (control.count_input >= sealed.inputs.size() ||
           !publication_matches_resolved(control.count,
                                         sealed.inputs[control.count_input]))) {
        return Status::fail(Reason::BindingInvalid);
      }
      step.window = static_cast<std::uint16_t>(control_index + 1u);
    }

    const std::uint32_t control_index = plan.window_states[step_index];
    if (declared.nested == 0u && control_index != PipelineResourceUnassigned) {
      if (control_index >= plan.window_controls.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineWindowControl &control =
          plan.window_controls[control_index];
      if (control.count_input >= sealed.inputs.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineResolvedViewPlan &count =
          sealed.inputs[control.count_input];
      if (!publication_matches_resolved(control.count, count) ||
          control.maximum == 0u || control.tile == 0u ||
          control.tile > control.maximum ||
          control.final < PipelineWindow::first ||
          control.final > PipelineWindow::second ||
          (control.terminal != std::numeric_limits<std::uint32_t>::max() &&
           control.terminal_output >= sealed.outputs.size())) {
        return Status::fail(Reason::BindingInvalid);
      }
      if (declared.iteration == 0u) {
        if (initialized_windows[control_index]) {
          return Status::fail(Reason::PipelineInvalid);
        }
        state->windows.entries_[control_index].descriptor = PipelineWindow{
            .control = control,
            .first_step = step_index,
            .recurrent_output_count =
                static_cast<std::uint32_t>(sealed.physical_sources.size()),
        };
        initialized_windows[control_index] = true;
      } else {
        if (!initialized_windows[control_index] ||
            state->windows[control_index].recurrent_output_count !=
                sealed.physical_sources.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
      step.window = static_cast<std::uint16_t>(control_index + 1u);
    }

    if (state->profile != nullptr) {
      PipelineStepProfile &profile = state->profile->steps[step_index];
      profile.index = declared.logical_step;
      profile.iteration = declared.iteration;
      profile.iteration_bound = declared.iteration_bound;
      profile.nested_phase = pipeline_nested_phase(declared.route);
      if (declared.route == PipelineRoute::NestedSeed) {
        profile.outer_window = declared.iteration;
      } else if (declared.route == PipelineRoute::NestedAction) {
        profile.inner_iteration = declared.iteration;
      }
      if (declared.nested != 0u) {
        const PipelineFrozenNestedWindow &nested =
            frozen.nested_windows[declared.nested - 1u];
        profile.outer_window_bound = nested.shape.outer_bound();
        profile.inner_iteration_bound = nested.shape.inner_bound();
      }
      profile.program = program->graph_info.fingerprint;
    }

    hash.number(step_index);
    hash.number(declared.logical_step);
    hash.number(declared.iteration);
    hash.number(declared.iteration_bound);
    const PipelineWindowControl *const identity_control =
        control_index == PipelineResourceUnassigned
            ? nullptr
            : &plan.window_controls[control_index];
    // Fingerprint v3 assigns these four slots to ordinary authored-step window
    // fields. Nested steps serialize their default values here and emit their
    // window identity once more in the nested-begin block below. Source every
    // live value from the one sealed state control.
    const PipelineWindowControl *const ordinary_control =
        declared.nested == 0u ? identity_control : nullptr;
    hash.number(ordinary_control == nullptr ? 0u : ordinary_control->maximum);
    hash.number(ordinary_control == nullptr ? 0u : ordinary_control->tile);
    hash.number(ordinary_control == nullptr ||
                        ordinary_control->terminal ==
                            std::numeric_limits<std::uint32_t>::max()
                    ? NoWindowTerminal
                    : ordinary_control->terminal);
    hash.number(ordinary_control == nullptr ? 1u : ordinary_control->expected);
    hash.number(declared.nested);
    hash.byte(static_cast<std::uint8_t>(declared.route));
    hash.byte(static_cast<std::uint8_t>(declared.writes_each_iteration));
    if (declared.nested != 0u) {
      const PipelineFrozenNestedWindow &nested =
          frozen.nested_windows[declared.nested - 1u];
      if (step_index == nested.shape.first()) {
        hash.number(identity_control->maximum);
        hash.number(identity_control->tile);
        hash.number(nested.shape.seed_count());
        hash.number(nested.shape.action_count());
        hash.number(nested.recurrent_output_count);
        hash.number(identity_control->terminal ==
                            std::numeric_limits<std::uint32_t>::max()
                        ? NoWindowTerminal
                        : identity_control->terminal);
        hash.number(identity_control->expected);
      }
    }
    hash.number(program->graph_info.fingerprint.hi);
    hash.number(program->graph_info.fingerprint.lo);
    hash.number(sealed.inputs.size());
    for (std::size_t index = 0u; index < sealed.inputs.size(); ++index) {
      const PipelineResolvedViewPlan &view = sealed.inputs[index];
      auto admitted = admit_resolved_view(
          plan, *state, view, program->input_types[index],
          program->input_sizes[index], program->input_formats[index],
          ResourceAccess::Read);
      if (!admitted) {
        return Status::fail(admitted.reason());
      }
      hash.byte(static_cast<std::uint8_t>(PipelineAccess::Read));
      hash.number(index);
      hash.number(*admitted);
      hash.number(static_cast<std::uint64_t>(program->input_types[index]));
      hash.number(program->input_sizes[index]);
      hash.number(view.offset);
      hash.number(view.count);
      hash.number(view.stride);
      hash.number(view.element_bytes);
      hash.number(view.alignment);
      hash.format(program->input_formats[index]);
    }

    hash.number(sealed.outputs.size());
    for (std::size_t index = 0u; index < sealed.outputs.size(); ++index) {
      const PipelineResolvedOutputPlan &output = sealed.outputs[index];
      if (output.physical >= program->output_types.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      auto admitted = admit_resolved_view(
          plan, *state, output.view, program->output_types[output.physical],
          program->output_sizes[output.physical],
          program->output_formats[output.physical], ResourceAccess::Write);
      if (!admitted) {
        return Status::fail(admitted.reason());
      }
      PipelineResource &output_resource = state->resources[*admitted];
      if (!output.hidden &&
          output_resource.output == PipelineResource::no_output) {
        output_resource.output = 0u;
        ++output_count;
      }
      hash.byte(static_cast<std::uint8_t>(PipelineAccess::Write));
      hash.number(index);
      hash.number(output.physical);
      hash.number(*admitted);
      hash.number(
          static_cast<std::uint64_t>(program->output_types[output.physical]));
      hash.number(program->output_sizes[output.physical]);
      hash.number(output.view.offset);
      hash.number(output.view.count);
      hash.number(output.view.stride);
      hash.number(output.view.element_bytes);
      hash.number(output.view.alignment);
      hash.byte(static_cast<std::uint8_t>(output.hidden));
      hash.format(program->output_formats[output.physical]);
    }
    for (std::size_t physical = 0u; physical < sealed.physical_sources.size();
         ++physical) {
      const PipelineResolvedViewPlan *view =
          physical_output_view(sealed, physical);
      if (view == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
    const Status aliases = validate_step_aliases(plan, sealed);
    if (!aliases) {
      return aliases;
    }
    step.writes = !sealed.physical_sources.empty();
  }

  if (std::find(initialized_windows.begin(), initialized_windows.end(),
                false) != initialized_windows.end()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!state->windows.empty()) {
    static_assert(PipelineRouteCapacity <=
                  std::numeric_limits<std::uint16_t>::max());
    state->window_rank.resize(state->steps.size() + 1u);
    for (std::size_t index = 0u; index < state->steps.size(); ++index) {
      state->window_rank[index + 1u] = static_cast<std::uint16_t>(
          state->window_rank[index] +
          (state->steps[index].window != 0u &&
                   state->steps[index].route == PipelineRoute::Ordinary
               ? 1u
               : 0u));
    }
  }

  return Status::success();
}

} // namespace rund::compute::detail
