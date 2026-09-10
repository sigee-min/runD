#include "internal.hpp"

#include "../../../program/output.hpp"
#include "../../../size.hpp"
#include "../../../type.hpp"

#include <rund/compute/pipeline.hpp>
#include <rund/compute/resource/plan.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail {

bool validate_window_request(PipelineBuildState &build,
                             const WindowAssemblyInput &input,
                             WindowAssemblyCounts &counts) {
  if (build.sealed || has_seed(build)) {
    build.failure = Reason::PipelineInvalid;
    return false;
  }
  const std::shared_ptr<ProgramState> &seed = *input.seed;
  const std::shared_ptr<ProgramState> &action = *input.action;
  const std::shared_ptr<ProgramState> &fold = *input.fold;
  const ResourceView &resident = *input.resident;
  if (seed == nullptr || fold == nullptr ||
      ((input.inner == 0u) != (action == nullptr))) {
    build.failure = Reason::ProgramInvalid;
    return false;
  }

  node::accel::detail::NestedTemplateShape nested_shape{};
  if (input.maximum == 0u || input.tile == 0u || input.tile > input.maximum ||
      input.maximum > std::numeric_limits<std::uint32_t>::max() ||
      input.inner > PipelineInnerIterationCapacity ||
      input.inner > std::numeric_limits<std::uint32_t>::max() ||
      !node::accel::detail::ProveNestedTemplateShape(
          build.steps.size(), static_cast<std::uint32_t>(input.maximum),
          static_cast<std::uint32_t>(input.tile),
          static_cast<std::uint32_t>(input.inner), nested_shape)) {
    build.failure = Reason::PipelineCapacity;
    return false;
  }
  const std::size_t seed_output_count =
      output_count(seed->output_aliases, seed->output_types.size());
  const std::size_t action_output_count =
      action == nullptr
          ? 0u
          : output_count(action->output_aliases, action->output_types.size());
  const std::size_t fold_output_count =
      output_count(fold->output_aliases, fold->output_types.size());
  const std::size_t recurrent_count = input.final_outputs.size();
  const std::size_t window_count = input.window_outputs.size();
  if (nested_shape.outer_bound() > PipelineIterationCapacity ||
      seed_output_count == 0u || recurrent_count == 0u ||
      fold_output_count != recurrent_count + window_count ||
      input.inputs.size() < recurrent_count ||
      seed->input_types.size() != input.inputs.size() - recurrent_count + 2u ||
      (action != nullptr &&
       (action_output_count == 0u || action_output_count > seed_output_count ||
        seed_output_count != action->input_types.size())) ||
      fold->input_types.size() != recurrent_count + seed_output_count ||
      (input.terminal != NoWindowTerminal &&
       (input.terminal >= recurrent_count ||
        fold->output_types[input.terminal] != Type::U32 ||
        fold->output_sizes[input.terminal] != 1u)) ||
      build.logical_step_count >= PipelineStepCapacity) {
    build.failure = Reason::PipelineCapacity;
    return false;
  }
  if (!seed->output_aliases.empty() ||
      (action != nullptr && !action->output_aliases.empty()) ||
      !fold->output_aliases.empty()) {
    build.failure = Reason::BindingAliasUnsupported;
    return false;
  }
  const std::uint64_t route_count = nested_shape.compact_entry_count();
  if (route_count > PipelineRouteCapacity ||
      build.steps.size() >
          PipelineRouteCapacity - static_cast<std::size_t>(route_count) ||
      seed->input_types.size() + seed_output_count > PipelineLeafCapacity ||
      (action != nullptr && action->input_types.size() + action_output_count >
                                PipelineLeafCapacity) ||
      fold->input_types.size() + fold_output_count > PipelineLeafCapacity) {
    build.failure = Reason::PipelineCapacity;
    return false;
  }
  if (resident.access != ResourceAccess::Read || resident.type != Type::U32 ||
      resident.count != 1u || resident.element_bytes != sizeof(std::uint32_t) ||
      std::any_of(input.inputs.begin(), input.inputs.end(),
                  [](const ResourceView &view) {
                    return view.access != ResourceAccess::Read;
                  }) ||
      std::any_of(input.final_outputs.begin(), input.final_outputs.end(),
                  [](const ResourceView &view) {
                    return view.access != ResourceAccess::Write;
                  }) ||
      std::any_of(input.window_outputs.begin(), input.window_outputs.end(),
                  [](const ResourceView &view) {
                    return view.access != ResourceAccess::Write;
                  })) {
    build.failure = Reason::BindingInvalid;
    return false;
  }

  const auto same_slot = [](const Type left_type, const FixedFormat left_format,
                            const std::size_t left_count, const Type right_type,
                            const FixedFormat right_format,
                            const std::size_t right_count) noexcept {
    return left_type == right_type && left_format == right_format &&
           left_count == right_count;
  };
  const std::size_t seed_external_count = input.inputs.size() - recurrent_count;
  for (std::size_t index = 0u; index < seed_external_count; ++index) {
    const ResourceView &view = input.inputs[recurrent_count + index];
    if (!same_slot(view.type, view.format, view.count, seed->input_types[index],
                   seed->input_formats[index], seed->input_sizes[index])) {
      build.failure = Reason::ShapeMismatch;
      return false;
    }
  }
  const std::size_t count_input = seed->input_types.size() - 2u;
  if (seed->input_types[count_input] != Type::U32 ||
      seed->input_types[count_input + 1u] != Type::U32 ||
      seed->input_sizes[count_input] != 1u ||
      seed->input_sizes[count_input + 1u] != 1u) {
    build.failure = Reason::ShapeMismatch;
    return false;
  }
  for (std::size_t index = 0u; index < seed_output_count; ++index) {
    if ((action != nullptr &&
         !same_slot(seed->output_types[index], seed->output_formats[index],
                    seed->output_sizes[index], action->input_types[index],
                    action->input_formats[index],
                    action->input_sizes[index])) ||
        !same_slot(seed->output_types[index], seed->output_formats[index],
                   seed->output_sizes[index],
                   fold->input_types[recurrent_count + index],
                   fold->input_formats[recurrent_count + index],
                   fold->input_sizes[recurrent_count + index])) {
      build.failure = Reason::ShapeMismatch;
      return false;
    }
  }
  for (std::size_t index = 0u; action != nullptr && index < action_output_count;
       ++index) {
    if (!same_slot(action->output_types[index], action->output_formats[index],
                   action->output_sizes[index], action->input_types[index],
                   action->input_formats[index], action->input_sizes[index])) {
      build.failure = Reason::ShapeMismatch;
      return false;
    }
  }
  for (std::size_t index = 0u; index < recurrent_count; ++index) {
    if (!same_slot(input.inputs[index].type, input.inputs[index].format,
                   input.inputs[index].count, fold->input_types[index],
                   fold->input_formats[index], fold->input_sizes[index]) ||
        !same_slot(input.final_outputs[index].type,
                   input.final_outputs[index].format,
                   input.final_outputs[index].count, fold->output_types[index],
                   fold->output_formats[index], fold->output_sizes[index]) ||
        !same_slot(fold->output_types[index], fold->output_formats[index],
                   fold->output_sizes[index], fold->input_types[index],
                   fold->input_formats[index], fold->input_sizes[index])) {
      build.failure = Reason::ShapeMismatch;
      return false;
    }
  }
  for (std::size_t index = 0u; index < window_count; ++index) {
    const ResourceView &target = input.window_outputs[index];
    const std::size_t output = recurrent_count + index;
    if (!same_slot(target.type, target.format, input.tile,
                   fold->output_types[output], fold->output_formats[output],
                   fold->output_sizes[output]) ||
        target.count != input.maximum || target.stride != 1u) {
      build.failure = Reason::ShapeMismatch;
      return false;
    }
  }
  const auto overlaps = [](const ResourceView &left,
                           const ResourceView &right) {
    return intersects(bind(left), right);
  };
  for (std::size_t index = 0u; index < window_count; ++index) {
    const ResourceView &target = input.window_outputs[index];
    for (std::size_t other = 0u; other < index; ++other) {
      if (target.buffer == input.window_outputs[other].buffer) {
        build.failure = Reason::BindingDuplicate;
        return false;
      }
      auto overlap = overlaps(target, input.window_outputs[other]);
      if (!overlap || *overlap) {
        build.failure =
            overlap ? Reason::BindingAliasUnsupported : overlap.reason();
        return false;
      }
    }
    auto resident_overlap = overlaps(target, resident);
    if (!resident_overlap || *resident_overlap) {
      build.failure = resident_overlap ? Reason::BindingAliasUnsupported
                                       : resident_overlap.reason();
      return false;
    }
    for (const ResourceView &view : input.inputs) {
      auto overlap = overlaps(target, view);
      if (!overlap || *overlap) {
        build.failure =
            overlap ? Reason::BindingAliasUnsupported : overlap.reason();
        return false;
      }
    }
    for (const ResourceView &view : input.final_outputs) {
      auto overlap = overlaps(target, view);
      if (!overlap || *overlap) {
        build.failure =
            overlap ? Reason::BindingAliasUnsupported : overlap.reason();
        return false;
      }
    }
  }

  std::size_t binding_count = 0u;
  const auto add_bindings = [&](const std::size_t count) {
    return size::add(binding_count, count, binding_count);
  };
  std::size_t seed_bindings = 0u;
  std::size_t action_bindings = 0u;
  std::size_t fold_bindings = 0u;
  if (!size::add(seed->input_types.size(), seed_output_count, seed_bindings) ||
      !size::multiply(seed_bindings, nested_shape.seed_count(),
                      seed_bindings) ||
      (action != nullptr && !size::add(action->input_types.size(),
                                       action_output_count, action_bindings)) ||
      !size::multiply(action_bindings, nested_shape.action_count(),
                      action_bindings) ||
      !size::add(fold->input_types.size(), fold_output_count, fold_bindings) ||
      !size::multiply(fold_bindings, nested_shape.fold_count(),
                      fold_bindings) ||
      !add_bindings(seed_bindings) || !add_bindings(action_bindings) ||
      !add_bindings(fold_bindings) ||
      build.binding_count > PipelineRouteBindingCapacity ||
      binding_count > PipelineRouteBindingCapacity - build.binding_count) {
    build.failure = Reason::PipelineCapacity;
    return false;
  }

  counts.nested_shape = nested_shape;
  counts.seed_output_count = seed_output_count;
  counts.action_output_count = action_output_count;
  counts.fold_output_count = fold_output_count;
  counts.recurrent_count = recurrent_count;
  counts.window_count = window_count;
  counts.seed_external_count = seed_external_count;
  counts.binding_count = binding_count;
  return true;
}

} // namespace rund::compute::detail
