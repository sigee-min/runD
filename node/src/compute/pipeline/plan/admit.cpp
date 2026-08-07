#include "prepare.hpp"
#include "publication.hpp"

#include "../../type.hpp"
#include "../claim.hpp"
#include "../local.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool publication_matches_resolved(
    const PipelinePublicationViewPlan &publication,
    const PipelineResolvedViewPlan &resolved) noexcept {
  return publication.identity.resource_ordinal == resolved.resource &&
         publication.identity.offset_bytes == resolved.offset_bytes &&
         publication.identity.count == resolved.count &&
         publication.identity.stride_bytes == resolved.stride_bytes &&
         publication.identity.element_bytes == resolved.element_bytes &&
         publication.type == resolved.declared_type &&
         publication.format == resolved.declared_format &&
         publication.offset == resolved.offset &&
         publication.stride == resolved.stride &&
         publication.alignment == resolved.alignment;
}

[[nodiscard]] Result<std::uint32_t>
admit_resolved_view(const PipelineMemoryPlan &plan, const PipelineState &state,
                    const PipelineResolvedViewPlan &view, const Type slot_type,
                    const std::size_t slot_count, const FixedFormat slot_format,
                    const ResourceAccess expected_access) noexcept {
  if (view.resource >= plan.resources.size() ||
      view.resource >= state.resources.size()) {
    return Result<std::uint32_t>::fail(Reason::PipelineInvalid);
  }
  const PipelineResolvedResourcePlan &planned = plan.resources[view.resource];
  const PipelineResource &resource = state.resources[view.resource];
  const std::shared_ptr<BufferState> &owner = resource.buffer;
  const bool external =
      std::holds_alternative<PipelineExternalResourcePlan>(planned.locator);
  if (owner == nullptr) {
    return Result<std::uint32_t>::fail(Reason::BindingInvalid);
  }
  const Status device = validate_pipeline_resource_device(state, resource);
  if (!device) {
    return Result<std::uint32_t>::fail(device.reason());
  }
  if (!valid_type(slot_type) || view.declared_type != slot_type ||
      planned.type != slot_type || resource.type != slot_type ||
      owner->type != slot_type) {
    return Result<std::uint32_t>::fail(Reason::BindingTypeMismatch);
  }
  if (!typed_format_matches(slot_type, view.declared_format, slot_format) ||
      planned.format != slot_format || resource.format != slot_format) {
    return Result<std::uint32_t>::fail(valid_format(slot_type, slot_format)
                                           ? Reason::FixedFormatMismatch
                                           : Reason::FixedFormatInvalid);
  }
  const std::size_t element_bytes = type_bytes(slot_type);
  if (view.declared_access != expected_access || element_bytes == 0u ||
      view.element_bytes != element_bytes || view.count != slot_count ||
      view.stride == 0u || view.alignment == 0u ||
      (view.alignment & (view.alignment - 1u)) != 0u || view.alignment > 64u ||
      view.offset_bytes % view.alignment != 0u ||
      view.declared_backing_bytes != planned.bytes ||
      planned.count != owner->count || planned.bytes != owner->bytes ||
      (external && planned.physical_bytes != owner->physical_bytes) ||
      owner->physical_bytes < owner->bytes ||
      view.offset_bytes > planned.bytes ||
      (view.count != 0u &&
       view.span_bytes > planned.bytes - view.offset_bytes)) {
    return Result<std::uint32_t>::fail(view.declared_access != expected_access
                                           ? Reason::BindingInvalid
                                           : Reason::ShapeMismatch);
  }
  return Result<std::uint32_t>::success(view.resource);
}

[[nodiscard]] PipelineResolvedViewPlan const *
physical_output_view(const PipelineStepResourcePlan &step,
                     const std::size_t physical) noexcept {
  if (physical >= step.physical_sources.size()) {
    return nullptr;
  }
  const std::uint32_t logical = step.physical_sources[physical];
  return logical < step.outputs.size() ? &step.outputs[logical].view : nullptr;
}

[[nodiscard]] Result<bool> resolved_views_intersect(
    const PipelineMemoryPlan &plan, const PipelineResolvedViewPlan &left,
    const resource::AccessMode left_mode, const PipelineResolvedViewPlan &right,
    const resource::AccessMode right_mode) noexcept {
  if (left.resource >= plan.resources.size() ||
      right.resource >= plan.resources.size()) {
    return Result<bool>::fail(Reason::PipelineInvalid);
  }
  const auto shape = [&](const PipelineResolvedViewPlan &view) {
    return resource::Resource{.id = view.resource + 1u,
                              .bytes = plan.resources[view.resource].bytes,
                              .alias_group = view.resource + 1u};
  };
  const auto access = [](const PipelineResolvedViewPlan &view,
                         const resource::AccessMode mode) {
    return resource::Access{.resource = view.resource + 1u,
                            .mode = mode,
                            .offset_bytes = view.offset_bytes,
                            .element_bytes = view.element_bytes,
                            .element_count = view.count,
                            .stride_bytes = view.stride_bytes};
  };
  return resource::intersects(shape(left), access(left, left_mode),
                              shape(right), access(right, right_mode));
}

[[nodiscard]] Status
validate_step_aliases(const PipelineMemoryPlan &plan,
                      const PipelineStepResourcePlan &step) noexcept {
  for (const PipelineResolvedViewPlan &input : step.inputs) {
    for (std::size_t physical = 0u; physical < step.physical_sources.size();
         ++physical) {
      const PipelineResolvedViewPlan *const output =
          physical_output_view(step, physical);
      if (output == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (input.resource != output->resource) {
        continue;
      }
      auto overlap =
          resolved_views_intersect(plan, input, resource::AccessMode::Read,
                                   *output, resource::AccessMode::Write);
      if (!overlap || *overlap) {
        return Status::fail(overlap ? Reason::BindingAliasUnsupported
                                    : overlap.reason());
      }
    }
  }
  for (std::size_t left = 0u; left < step.physical_sources.size(); ++left) {
    const PipelineResolvedViewPlan *const left_view =
        physical_output_view(step, left);
    if (left_view == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    for (std::size_t right = left + 1u; right < step.physical_sources.size();
         ++right) {
      const PipelineResolvedViewPlan *const right_view =
          physical_output_view(step, right);
      if (right_view == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (left_view->resource != right_view->resource) {
        continue;
      }
      auto overlap = resolved_views_intersect(
          plan, *left_view, resource::AccessMode::Write, *right_view,
          resource::AccessMode::Write);
      if (!overlap || *overlap) {
        return Status::fail(overlap ? Reason::BindingDuplicate
                                    : overlap.reason());
      }
    }
  }
  return Status::success();
}

} // namespace

Status admit_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                      PipelinePrepare &prepare) {
  if (build->memory == nullptr || build->memory->frozen == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineMemoryPlan &plan = *build->memory;
  const PipelineBuildSnapshot &frozen = *plan.frozen;
  if (plan.window_states.size() != frozen.steps.size() ||
      plan.step_resources.size() != frozen.steps.size() ||
      frozen.commit != !plan.state_pair_resources.empty() ||
      build->materialized_resources.size() != plan.resources.size() ||
      plan.resources.size() != plan.hazards.lifetimes.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t resource_count = plan.resources.size();
  if (resource_count > PipelineResourceCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }

  auto state = std::make_shared<PipelineState>();
  state->device = frozen.device;
  state->publication = std::make_shared<PipelinePublicationState>();
  state->publication->device = frozen.device;
  state->sealed_repetitions = frozen.sealed_repetitions;
  state->plan = plan.summary;
  state->steps.resize(frozen.steps.size());
  if (plan.window_controls.size() > std::numeric_limits<std::uint16_t>::max()) {
    return Status::fail(Reason::PipelineCapacity);
  }
  state->windows.resize(plan.window_controls.size());
  std::vector<bool> initialized_windows(plan.window_controls.size(), false);
  state->resources.reserve(resource_count);
  state->barriers.resize(frozen.steps.size());
  if (frozen.profile == PipelineProfile::Steps) {
    state->profile = std::make_unique<PipelineProfileState>();
    state->profile->steps.resize(frozen.steps.size());
    state->profile->started_ns.resize(frozen.steps.size());
    state->profile->started.resize(frozen.steps.size());
  }

  for (std::size_t ordinal = 0u; ordinal < resource_count; ++ordinal) {
    const PipelineResolvedResourcePlan &planned = plan.resources[ordinal];
    const std::shared_ptr<BufferState> &owner =
        build->materialized_resources[ordinal];
    const bool owned =
        std::holds_alternative<PipelineInternalResourcePlan>(planned.locator);
    if (owner == nullptr ||
        planned.count > std::numeric_limits<std::size_t>::max() ||
        planned.bytes > std::numeric_limits<std::size_t>::max() ||
        (planned.first_write != resource::NoNode &&
         planned.first_write >= frozen.steps.size())) {
      return Status::fail(owner == nullptr ? Reason::BindingInvalid
                                           : Reason::PipelineInvalid);
    }
    state->resources.push_back(PipelineResource{
        .buffer = owner,
        .type = planned.type,
        .format = planned.format,
        .count = static_cast<std::size_t>(planned.count),
        .bytes = static_cast<std::size_t>(planned.bytes),
        .output = PipelineResource::no_output,
        .first_write = planned.first_write,
        .owned = owned,
        .terminal_publish = false,
    });
  }

  PipelineHash hash{};
  hash.number(frozen.sealed_repetitions);
  hash.number(frozen.steps.size());
  std::size_t observed_bindings = 0u;
  std::uint64_t status_entry_count = 0u;
  const std::size_t binding_capacity = frozen.nested_windows.empty()
                                           ? PipelineBindingCapacity
                                           : PipelineRouteBindingCapacity;
  std::size_t output_count = 0u;

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
        state->windows[control_index] = PipelineWindow{
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
        state->windows[control_index] = PipelineWindow{
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
    // Fingerprint v3 assigned these four slots to ordinary authored-step
    // window fields. Nested steps historically serialized their default values
    // here and emitted their window identity once more in the nested-begin
    // block below. Preserve that byte contract while sourcing every live value
    // from the one sealed state control.
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
  const auto exact_resource =
      [&](const PipelinePublicationViewPlan &view) -> PipelineResource * {
    const PipelinePublicationViewIdentity &identity = view.identity;
    if (identity.resource_ordinal >= state->resources.size() ||
        identity.resource_ordinal >= plan.resources.size()) {
      return nullptr;
    }
    PipelineResource &resource = state->resources[identity.resource_ordinal];
    return resource.buffer != nullptr && resource.type == view.type &&
                   resource.format == view.format &&
                   resource.buffer->type == view.type &&
                   resource.bytes == identity.backing_bytes &&
                   resource.buffer->bytes == identity.backing_bytes &&
                   identity.element_bytes != 0u &&
                   identity.stride_bytes != 0u &&
                   identity.offset_bytes <= identity.backing_bytes &&
                   (identity.count == 0u ||
                    (identity.element_bytes <=
                         identity.backing_bytes - identity.offset_bytes &&
                     identity.count - 1u <=
                         (identity.backing_bytes - identity.offset_bytes -
                          identity.element_bytes) /
                             identity.stride_bytes))
               ? &resource
               : nullptr;
  };

  state->publications.reserve(plan.publications.size());
  hash.number(plan.publications.size());
  for (const PipelinePublicationPlan &planned : plan.publications) {
    const auto *window = std::get_if<PipelineWindowPublicationPlan>(&planned);
    const auto *terminal =
        std::get_if<PipelineTerminalPublicationPlan>(&planned);
    const std::uint32_t planned_state =
        window != nullptr ? window->state : terminal->state;
    const std::uint32_t physical =
        window != nullptr ? window->output.value : terminal->output.value;
    if (planned_state >= state->windows.size() ||
        physical >= PipelineLeafCapacity ||
        physical > std::numeric_limits<std::uint16_t>::max()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineWindow &publication_window = state->windows[planned_state];
    const PipelineWindowControl &control = publication_window.control;
    const PipelinePublicationTargetPlan &target_plan =
        pipeline_publication_target(planned);
    const std::uint32_t target = target_plan.view.identity.resource_ordinal;
    if (target >= plan.resources.size() || target >= state->resources.size() ||
        !std::holds_alternative<PipelineExternalResourcePlan>(
            plan.resources[target].locator)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const Status target_device =
        validate_pipeline_resource_device(*state, state->resources[target]);
    if (!target_device) {
      return target_device;
    }
    if (exact_resource(target_plan.view) == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (window != nullptr) {
      if (exact_resource(window->source) == nullptr ||
          exact_resource(control.count) == nullptr || control.maximum == 0u ||
          control.tile == 0u || control.tile > control.maximum) {
        return Status::fail(Reason::PipelineInvalid);
      }
    } else {
      if (terminal == nullptr || control.final < PipelineWindow::first ||
          control.final > PipelineWindow::second ||
          control.final >= terminal->sources.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      for (const PipelinePublicationViewPlan &bank : terminal->sources) {
        if (exact_resource(bank) == nullptr) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
    }
    PipelineResource &target_resource = state->resources[target];
    if (!plan.resources[target].output ||
        target_resource.output != PipelineResource::no_output) {
      return Status::fail(target_resource.output != PipelineResource::no_output
                              ? Reason::BindingDuplicate
                              : Reason::PipelineInvalid);
    }
    target_resource.output = 0u;
    target_resource.terminal_publish = window == nullptr;
    ++output_count;
    state->publications.push_back(planned);
    if (!mix_pipeline_publication_public_identity(hash, planned, control)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  for (std::size_t ordinal = 0u; ordinal < plan.resources.size(); ++ordinal) {
    const PipelineResolvedResourcePlan &planned = plan.resources[ordinal];
    const PipelineResource &admitted = state->resources[ordinal];
    if (planned.output != (admitted.output != PipelineResource::no_output) ||
        (planned.output &&
         planned.terminal_publish != admitted.terminal_publish)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }

  state->transactional = frozen.commit;
  state->publication->state_pairs.reserve(plan.state_pair_resources.size());
  hash.number(plan.state_pair_resources.size());
  for (const PipelineStatePairResourcePlan &pair : plan.state_pair_resources) {
    if (pair.published.resource >= plan.resources.size() ||
        pair.pending.resource >= plan.resources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineResolvedResourcePlan &published_plan =
        plan.resources[pair.published.resource];
    const PipelineResolvedResourcePlan &pending_plan =
        plan.resources[pair.pending.resource];
    auto published =
        admit_resolved_view(plan, *state, pair.published, published_plan.type,
                            static_cast<std::size_t>(published_plan.count),
                            published_plan.format, ResourceAccess::Read);
    auto pending =
        admit_resolved_view(plan, *state, pair.pending, pending_plan.type,
                            static_cast<std::size_t>(pending_plan.count),
                            pending_plan.format, ResourceAccess::Write);
    if (!published || !pending || *published == *pending) {
      return Status::fail(!published ? published.reason()
                          : !pending ? pending.reason()
                                     : Reason::PipelineInvalid);
    }
    PipelineResource &published_resource = state->resources[*published];
    PipelineResource &pending_resource = state->resources[*pending];
    if (*published != pair.published.resource ||
        *pending != pair.pending.resource) {
      return Status::fail(Reason::PipelineInvalid);
    }
    // A state pair shares one typed storage schema, not one Program arithmetic
    // policy. Fixed rounding/overflow/approximation remain canonical on each
    // resource's producing/consuming Program; parity only swaps equal-width
    // storage owners, as the public state contract has always allowed.
    if (pair.published.offset != 0u || pair.published.stride != 1u ||
        pair.published.count != published_plan.count ||
        pair.pending.offset != 0u || pair.pending.stride != 1u ||
        pair.pending.count != pending_plan.count ||
        published_resource.type != pending_resource.type ||
        published_resource.count != pending_resource.count ||
        published_resource.bytes != pending_resource.bytes ||
        published_plan.output ||
        pair.pending_first_full_write == resource::NoNode ||
        pair.pending_first_input <= pair.pending_first_full_write ||
        published_resource.partner != PipelineResource::no_output ||
        pending_resource.partner != PipelineResource::no_output) {
      return Status::fail(Reason::PipelineInvalid);
    }
    published_resource.partner = *pending;
    pending_resource.partner = *published;
    state->publication->state_pairs.push_back(PipelineStatePair{
        .first = build->materialized_resources[*published],
        .second = build->materialized_resources[*pending],
        .type = published_resource.type,
        .format = published_resource.format,
        .count = published_resource.count,
        .bytes = published_resource.bytes,
    });
    hash.number(*published);
    hash.number(*pending);
    hash.number(static_cast<std::uint64_t>(published_resource.type));
    hash.format(published_resource.format);
    hash.number(published_resource.count);
    hash.number(published_resource.bytes);
  }

  if (state->resources.size() != resource_count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  prepare.state = std::move(state);
  prepare.hash = hash;
  prepare.output_count = output_count;
  prepare.status_count = status_entry_count;
  return Status::success();
}

} // namespace rund::compute::detail
