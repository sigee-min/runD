#include "prepare.hpp"

#include "../../size.hpp"
#include "../../type.hpp"
#include "../local.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rund::compute::detail {

Status admit_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                      PipelinePrepare &prepare) {
  auto state = std::make_shared<PipelineState>();
  state->device = build->device;
  state->sealed_repetitions = build->sealed_repetitions;
  if (build->memory == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state->plan = build->memory->summary;
  state->steps.resize(build->steps.size());
  state->windows.reserve(build->logical_step_count);
  if (build->profile == PipelineProfile::Steps) {
    state->profile = std::make_unique<PipelineProfileState>();
    state->profile->steps.resize(build->steps.size());
    state->profile->started_ns.resize(build->steps.size());
    state->profile->started.resize(build->steps.size());
  }
  std::vector<PipelinePlanStep> plan_steps(build->steps.size());
  std::vector<PipelineResourceAdmission> resource_admissions;
  resource_admissions.reserve(
      std::min(build->binding_count, PipelineResourceCapacity));
  std::vector<PhysicalOutputProjection> physical_outputs(build->steps.size());
  for (PhysicalOutputProjection &projection : physical_outputs) {
    projection.sources.fill(PhysicalOutputProjection::unassigned);
  }
  std::vector<std::uint16_t> nested_windows(build->nested_windows.size());
  state->resources.reserve(
      std::min(build->binding_count, PipelineResourceCapacity));
  state->barriers.resize(build->steps.size());

  std::unordered_map<const BufferState *, std::uint32_t> ordinals;
  ordinals.reserve(std::min(build->binding_count, PipelineResourceCapacity));
  PipelineHash hash{};
  hash.number(build->sealed_repetitions);
  hash.number(build->steps.size());
  std::size_t observed_bindings = 0u;
  std::size_t output_count = 0u;
  std::uint64_t status_entry_count = 0u;
  const std::size_t binding_capacity = build->nested_windows.empty()
                                           ? PipelineBindingCapacity
                                           : PipelineRouteBindingCapacity;

  const auto admit =
      [&](const PipelineBinding &binding, const Type slot_type,
          const std::size_t slot_count,
          const FixedFormat slot_format) -> Result<std::uint32_t> {
    const bool owned = binding.owner != PipelineBinding::external;
    if (binding.buffer == nullptr) {
      return Result<std::uint32_t>::fail(Reason::BindingInvalid);
    }
    if (binding.buffer->device != state->device) {
      return Result<std::uint32_t>::fail(Reason::BindingDeviceMismatch);
    }
    if (!valid_type(slot_type) || binding.type != slot_type ||
        binding.buffer->type != slot_type) {
      return Result<std::uint32_t>::fail(Reason::BindingTypeMismatch);
    }
    const std::size_t element_bytes = type_bytes(slot_type);
    if (element_bytes == 0u || !size::multiply(slot_count, element_bytes)) {
      return Result<std::uint32_t>::fail(Reason::ShapeMismatch);
    }
    std::size_t distance = 0u;
    std::size_t last = binding.offset;
    const bool footprint_overflow =
        binding.count != 0u &&
        (binding.stride == 0u ||
         !size::multiply(binding.count - 1u, binding.stride, distance) ||
         !size::add(binding.offset, distance, last));
    std::size_t buffer_bytes = 0u;
    std::size_t offset_bytes = 0u;
    const bool valid_buffer_bytes =
        size::multiply(binding.buffer->count, element_bytes, buffer_bytes);
    const bool valid_offset =
        size::multiply(binding.offset, element_bytes, offset_bytes);
    if (binding.count != slot_count || binding.element_bytes != element_bytes ||
        binding.alignment == 0u ||
        (binding.alignment & (binding.alignment - 1u)) != 0u ||
        binding.alignment > 64u || footprint_overflow ||
        (binding.count != 0u && last >= binding.buffer->count) ||
        !valid_buffer_bytes || binding.buffer->bytes != buffer_bytes ||
        binding.backing_bytes != binding.buffer->bytes ||
        binding.buffer->physical_bytes < binding.buffer->bytes ||
        !valid_offset || offset_bytes % binding.alignment != 0u) {
      return Result<std::uint32_t>::fail(Reason::ShapeMismatch);
    }
    if (!typed_format_matches(slot_type, binding.format, slot_format)) {
      return Result<std::uint32_t>::fail(valid_format(slot_type, slot_format)
                                             ? Reason::FixedFormatMismatch
                                             : Reason::FixedFormatInvalid);
    }

    const auto found = ordinals.find(binding.buffer.get());
    if (found != ordinals.end()) {
      PipelineResource &resource = state->resources[found->second];
      if (resource.type != slot_type) {
        return Result<std::uint32_t>::fail(Reason::BindingTypeMismatch);
      }
      if (resource.count != binding.buffer->count ||
          resource.bytes != binding.buffer->bytes) {
        return Result<std::uint32_t>::fail(Reason::ShapeMismatch);
      }
      if (resource.format != slot_format) {
        return Result<std::uint32_t>::fail(Reason::FixedFormatMismatch);
      }
      if (resource.owned != owned) {
        return Result<std::uint32_t>::fail(Reason::PipelineInvalid);
      }
      return Result<std::uint32_t>::success(found->second);
    }
    if (state->resources.size() >= PipelineResourceCapacity) {
      return Result<std::uint32_t>::fail(Reason::PipelineCapacity);
    }
    const auto ordinal = static_cast<std::uint32_t>(state->resources.size());
    ordinals.emplace(binding.buffer.get(), ordinal);
    state->resources.push_back(PipelineResource{
        .buffer = binding.buffer,
        .type = slot_type,
        .format = slot_format,
        .count = binding.buffer->count,
        .bytes = binding.buffer->bytes,
        .owned = owned,
    });
    resource_admissions.emplace_back();
    return Result<std::uint32_t>::success(ordinal);
  };

  // Phase 1 owns every static check. No Job or backend preparation occurs
  // until all steps, projections, bindings, and policies are accepted.
  for (std::size_t step_index = 0u; step_index < build->steps.size();
       ++step_index) {
    const PipelineBuildStep &declared = build->steps[step_index];
    const ProgramState *const program = declared.program.get();
    if (program == nullptr || program->device != state->device) {
      return Status::fail(program == nullptr ? Reason::ProgramInvalid
                                             : Reason::BindingDeviceMismatch);
    }
    status_entry_count = ::rund::detail::counter::SaturatingAdd(
        status_entry_count, cpu_program_status_entries(*program));
    if (!valid_input_shape(*program) ||
        declared.inputs.size() != program->input_types.size()) {
      return Status::fail(Reason::BindingCountMismatch);
    }
    auto projection = project_outputs(declared);
    if (!projection) {
      return Status::fail(projection.reason());
    }
    if (declared.inputs.size() > PipelineLeafCapacity ||
        declared.outputs.size() >
            PipelineLeafCapacity - declared.inputs.size()) {
      return Status::fail(Reason::PipelineCapacity);
    }
    if (observed_bindings > binding_capacity ||
        declared.inputs.size() + declared.outputs.size() >
            binding_capacity - observed_bindings) {
      return Status::fail(Reason::PipelineCapacity);
    }
    observed_bindings += declared.inputs.size() + declared.outputs.size();

    PipelineStep &step = state->steps[step_index];
    PipelinePlanStep &planned_step = plan_steps[step_index];
    PhysicalOutputProjection &physical_output = physical_outputs[step_index];
    step.program = declared.program;
    step.logical_step = declared.logical_step;
    step.iteration = declared.iteration;
    step.iteration_bound = declared.iteration_bound;
    step.route = declared.route;
    step.writes_each_iteration = declared.writes_each_iteration;
    if (declared.nested != 0u) {
      const std::size_t nested_index =
          static_cast<std::size_t>(declared.nested - 1u);
      if (nested_index >= build->nested_windows.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineBuildNestedWindow &nested =
          build->nested_windows[nested_index];
      if (nested.begin >= nested.end || nested.end > build->steps.size() ||
          step_index < nested.begin || step_index >= nested.end ||
          nested.seed_first != nested.begin || nested.seed_count == 0u ||
          nested.action_count == 0u ||
          nested.action_first != nested.seed_first + nested.seed_count ||
          nested.fold_first != nested.action_first + nested.action_count ||
          nested.end != nested.fold_first + 3u || nested.maximum == 0u ||
          nested.tile == 0u || nested.tile > nested.maximum ||
          nested.count.buffer == nullptr || nested.count.type != Type::U32 ||
          nested.count.count != 1u ||
          nested.count.element_bytes != sizeof(std::uint32_t)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (step_index == nested.begin) {
        if (state->windows.size() >=
            std::numeric_limits<std::uint16_t>::max()) {
          return Status::fail(Reason::PipelineCapacity);
        }
        auto fold_projection = project_outputs(build->steps[nested.fold_first]);
        if (!fold_projection ||
            (nested.terminal != NoWindowTerminal &&
             nested.terminal >=
                 build->steps[nested.fold_first].outputs.size())) {
          return Status::fail(fold_projection ? Reason::PipelineInvalid
                                              : fold_projection.reason());
        }
        state->windows.push_back(PipelineWindow{
            .count = nested.count.buffer,
            .count_offset = nested.count.offset,
            .first_step = nested.fold_first,
            .maximum = static_cast<std::uint32_t>(nested.maximum),
            .tile = static_cast<std::uint32_t>(nested.tile),
            .terminal = nested.terminal == NoWindowTerminal
                            ? std::numeric_limits<std::uint32_t>::max()
                            : static_cast<std::uint32_t>(nested.terminal),
            .terminal_output =
                nested.terminal == NoWindowTerminal
                    ? 0u
                    : fold_projection->logical_to_physical[nested.terminal],
            .expected = nested.expected,
            .begin = nested.begin,
            .end = nested.end,
            .seed_first = nested.seed_first,
            .seed_count = nested.seed_count,
            .action_first = nested.action_first,
            .action_count = nested.action_count,
            .fold_first = nested.fold_first,
            .nested = true,
        });
        nested_windows[nested_index] =
            static_cast<std::uint16_t>(state->windows.size());
      }
      if (nested_windows[nested_index] == 0u) {
        return Status::fail(Reason::PipelineInvalid);
      }
      step.window = nested_windows[nested_index];
    }
    if (declared.window_tile != 0u) {
      if (declared.nested != 0u) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (declared.window_max == 0u ||
          declared.window_max > std::numeric_limits<std::uint32_t>::max() ||
          declared.window_tile > std::numeric_limits<std::uint32_t>::max() ||
          (declared.window_terminal != NoWindowTerminal &&
           (declared.window_terminal >= declared.outputs.size() ||
            declared.window_terminal >
                std::numeric_limits<std::uint32_t>::max())) ||
          declared.inputs.size() < 2u) {
        return Status::fail(Reason::PipelineCapacity);
      }
      const PipelineBinding &count =
          declared.inputs[declared.inputs.size() - 2u];
      if (count.buffer == nullptr || count.type != Type::U32 ||
          count.count != 1u || count.element_bytes != sizeof(std::uint32_t) ||
          count.stride == 0u) {
        return Status::fail(Reason::BindingInvalid);
      }
      if (declared.iteration == 0u) {
        if (state->windows.size() >=
            std::numeric_limits<std::uint16_t>::max()) {
          return Status::fail(Reason::PipelineCapacity);
        }
        state->windows.push_back(PipelineWindow{
            .count = count.buffer,
            .count_offset = count.offset,
            .first_step = step_index,
            .maximum = static_cast<std::uint32_t>(declared.window_max),
            .tile = static_cast<std::uint32_t>(declared.window_tile),
            .terminal =
                declared.window_terminal == NoWindowTerminal
                    ? std::numeric_limits<std::uint32_t>::max()
                    : static_cast<std::uint32_t>(declared.window_terminal),
            .terminal_output =
                declared.window_terminal == NoWindowTerminal
                    ? 0u
                    : projection->logical_to_physical[declared.window_terminal],
            .expected = declared.window_expected,
        });
        step.window = static_cast<std::uint16_t>(state->windows.size());
      } else {
        if (step_index == 0u || state->steps[step_index - 1u].window == 0u) {
          return Status::fail(Reason::PipelineInvalid);
        }
        step.window = state->steps[step_index - 1u].window;
        const PipelineWindow &window = state->windows[step.window - 1u];
        if (window.maximum != declared.window_max ||
            window.tile != declared.window_tile ||
            window.terminal != (declared.window_terminal == NoWindowTerminal
                                    ? std::numeric_limits<std::uint32_t>::max()
                                    : declared.window_terminal) ||
            window.expected != declared.window_expected ||
            window.count != count.buffer ||
            window.count_offset != count.offset) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
    }
    if (state->profile != nullptr) {
      PipelineStepProfile &profile = state->profile->steps[step_index];
      profile.index = declared.logical_step;
      profile.iteration = declared.iteration;
      profile.iteration_bound = declared.iteration_bound;
      if (declared.route == PipelineRoute::NestedSeed) {
        profile.outer_window = declared.iteration;
        profile.nested_phase = PipelineNestedPhase::Seed;
      } else if (declared.route == PipelineRoute::NestedAction) {
        profile.inner_iteration = declared.iteration;
        profile.nested_phase = PipelineNestedPhase::Action;
      } else if (declared.route == PipelineRoute::NestedFold) {
        profile.nested_phase = PipelineNestedPhase::Fold;
      }
      if (declared.nested != 0u) {
        const PipelineBuildNestedWindow &nested =
            build->nested_windows[declared.nested - 1u];
        profile.outer_window_bound =
            static_cast<std::uint32_t>(nested.seed_count);
        profile.inner_iteration_bound =
            static_cast<std::uint32_t>(nested.action_count);
      }
      profile.program = program->graph_info.fingerprint;
    }
    planned_step.inputs.resize(declared.inputs.size());
    planned_step.outputs.resize(projection->physical_count);
    hash.number(step_index);
    hash.number(declared.logical_step);
    hash.number(declared.iteration);
    hash.number(declared.iteration_bound);
    hash.number(declared.window_max);
    hash.number(declared.window_tile);
    hash.number(declared.window_terminal);
    hash.number(declared.window_expected);
    hash.number(declared.nested);
    hash.byte(static_cast<std::uint8_t>(declared.route));
    hash.byte(static_cast<std::uint8_t>(declared.writes_each_iteration));
    if (declared.nested != 0u) {
      const PipelineBuildNestedWindow &nested =
          build->nested_windows[declared.nested - 1u];
      if (step_index == nested.begin) {
        hash.number(nested.maximum);
        hash.number(nested.tile);
        hash.number(nested.seed_count);
        hash.number(nested.action_count);
        hash.number(nested.terminal);
        hash.number(nested.expected);
      }
    }
    hash.number(program->graph_info.fingerprint.hi);
    hash.number(program->graph_info.fingerprint.lo);
    hash.number(declared.inputs.size());
    for (std::size_t index = 0u; index < declared.inputs.size(); ++index) {
      auto ordinal =
          admit(declared.inputs[index], program->input_types[index],
                program->input_sizes[index], program->input_formats[index]);
      if (!ordinal) {
        return Status::fail(ordinal.reason());
      }
      planned_step.inputs[index] = *ordinal;
      PipelineResourceAdmission &admission = resource_admissions[*ordinal];
      admission.first_input_step = std::min(
          admission.first_input_step, static_cast<std::uint32_t>(step_index));
      hash.byte(static_cast<std::uint8_t>(PipelineAccess::Read));
      hash.number(index);
      hash.number(*ordinal);
      hash.number(static_cast<std::uint64_t>(program->input_types[index]));
      hash.number(program->input_sizes[index]);
      hash.number(declared.inputs[index].offset);
      hash.number(declared.inputs[index].count);
      hash.number(declared.inputs[index].stride);
      hash.number(declared.inputs[index].element_bytes);
      hash.number(declared.inputs[index].alignment);
      hash.format(program->input_formats[index]);
    }

    std::array<std::uint32_t, PipelineLeafCapacity> physical_ordinals{};
    physical_ordinals.fill(std::numeric_limits<std::uint32_t>::max());
    std::array<bool, PipelineLeafCapacity> physical_bound{};
    hash.number(declared.outputs.size());
    for (std::size_t index = 0u; index < declared.outputs.size(); ++index) {
      const std::uint32_t physical = projection->logical_to_physical[index];
      auto admitted = admit(
          declared.outputs[index], program->output_types[physical],
          program->output_sizes[physical], program->output_formats[physical]);
      if (!admitted) {
        return Status::fail(admitted.reason());
      }
      // project_outputs already proves that every logical alias of this
      // physical output names the same exact View. admit therefore returns
      // the same canonical owner ordinal for every repeated physical index.
      physical_ordinals[physical] = *admitted;
      if (!physical_bound[physical]) {
        physical_output.sources[physical] = static_cast<std::uint8_t>(index);
        physical_bound[physical] = true;
      }
      const std::uint32_t ordinal = physical_ordinals[physical];
      PipelineResource &output_resource = state->resources[ordinal];
      output_resource.first_write = std::min(
          output_resource.first_write, static_cast<std::uint32_t>(step_index));
      if (!declared.outputs[index].hidden &&
          output_resource.output == PipelineResource::no_output) {
        // Mark write membership now; the canonical output index is assigned
        // in resource order once Phase 1 is complete.
        output_resource.output = 0u;
        ++output_count;
      }
      if (declared.outputs[index].offset == 0u &&
          declared.outputs[index].stride == 1u &&
          declared.outputs[index].count ==
              declared.outputs[index].buffer->count) {
        PipelineResourceAdmission &admission = resource_admissions[ordinal];
        admission.first_full_output_step =
            std::min(admission.first_full_output_step,
                     static_cast<std::uint32_t>(step_index));
      }
      hash.byte(static_cast<std::uint8_t>(PipelineAccess::Write));
      hash.number(index);
      hash.number(physical);
      hash.number(ordinal);
      hash.number(static_cast<std::uint64_t>(program->output_types[physical]));
      hash.number(program->output_sizes[physical]);
      hash.number(declared.outputs[index].offset);
      hash.number(declared.outputs[index].count);
      hash.number(declared.outputs[index].stride);
      hash.number(declared.outputs[index].element_bytes);
      hash.number(declared.outputs[index].alignment);
      hash.byte(static_cast<std::uint8_t>(declared.outputs[index].hidden));
      hash.format(program->output_formats[physical]);
    }
    std::copy_n(physical_ordinals.begin(), projection->physical_count,
                planned_step.outputs.begin());
    step.writes = !planned_step.outputs.empty();

    const auto resource_descriptor = [&](const std::uint32_t ordinal) {
      const PipelineResource &value = state->resources[ordinal];
      return resource::Resource{
          .id = ordinal + 1u,
          .bytes = value.bytes,
          .alias_group = ordinal + 1u,
      };
    };
    const auto access_descriptor = [&](const PipelineBinding &binding,
                                       const std::uint32_t ordinal,
                                       const resource::AccessMode mode) {
      return resource::Access{
          .resource = ordinal + 1u,
          .mode = mode,
          .offset_bytes = binding.offset * binding.element_bytes,
          .element_bytes = binding.element_bytes,
          .element_count = binding.count,
          .stride_bytes = binding.stride * binding.element_bytes,
      };
    };
    for (std::size_t input = 0u; input < planned_step.inputs.size(); ++input) {
      for (std::size_t output = 0u; output < planned_step.outputs.size();
           ++output) {
        if (planned_step.inputs[input] != planned_step.outputs[output]) {
          continue;
        }
        const PipelineBinding &output_binding =
            declared.outputs[physical_output.sources[output]];
        auto overlap = resource::intersects(
            resource_descriptor(planned_step.inputs[input]),
            access_descriptor(declared.inputs[input],
                              planned_step.inputs[input],
                              resource::AccessMode::Read),
            resource_descriptor(planned_step.outputs[output]),
            access_descriptor(output_binding, planned_step.outputs[output],
                              resource::AccessMode::Write));
        if (!overlap) {
          return Status::fail(overlap.reason());
        }
        if (*overlap) {
          return Status::fail(Reason::BindingAliasUnsupported);
        }
      }
    }
    for (std::size_t left = 0u; left < planned_step.outputs.size(); ++left) {
      for (std::size_t right = left + 1u; right < planned_step.outputs.size();
           ++right) {
        if (planned_step.outputs[left] != planned_step.outputs[right]) {
          continue;
        }
        const PipelineBinding &left_binding =
            declared.outputs[physical_output.sources[left]];
        const PipelineBinding &right_binding =
            declared.outputs[physical_output.sources[right]];
        auto overlap = resource::intersects(
            resource_descriptor(planned_step.outputs[left]),
            access_descriptor(left_binding, planned_step.outputs[left],
                              resource::AccessMode::Write),
            resource_descriptor(planned_step.outputs[right]),
            access_descriptor(right_binding, planned_step.outputs[right],
                              resource::AccessMode::Write));
        if (!overlap) {
          return Status::fail(overlap.reason());
        }
        if (*overlap) {
          return Status::fail(Reason::BindingDuplicate);
        }
      }
    }
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
  if (observed_bindings != build->binding_count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state->publications.reserve(build->publications.size());
  hash.number(build->publications.size());
  for (const PipelineBuildPublish &declared : build->publications) {
    if (declared.step >= state->steps.size() ||
        state->steps[declared.step].window == 0u ||
        declared.output >= PipelineLeafCapacity) {
      return Status::fail(Reason::PipelineInvalid);
    }
    auto projection = project_outputs(build->steps[declared.step]);
    if (!projection ||
        declared.output >= build->steps[declared.step].outputs.size()) {
      return Status::fail(projection ? Reason::PipelineInvalid
                                     : projection.reason());
    }
    const std::uint32_t physical =
        projection->logical_to_physical[declared.output];
    if (physical >= projection->physical_count ||
        physical > std::numeric_limits<std::uint16_t>::max()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const auto source = ordinals.find(declared.source.buffer.get());
    if (source == ordinals.end()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::uint32_t source_ordinal = source->second;
    const Type source_type = state->resources[source_ordinal].type;
    const FixedFormat source_format = state->resources[source_ordinal].format;
    auto target = admit(declared.target, source_type, declared.source.count,
                        source_format);
    if (!target) {
      return Status::fail(target.reason());
    }
    if (source_ordinal == *target ||
        declared.source.type != declared.target.type ||
        declared.source.format != declared.target.format ||
        declared.source.count != declared.target.count ||
        declared.source.stride != 1u || declared.source.offset != 0u ||
        declared.source.element_bytes != declared.target.element_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    PipelineResource &target_resource = state->resources[*target];
    if (target_resource.output != PipelineResource::no_output) {
      return Status::fail(Reason::BindingDuplicate);
    }
    target_resource.output = 0u;
    target_resource.terminal_publish = true;
    target_resource.first_write = static_cast<std::uint32_t>(
        build->steps.empty() ? 0u : build->steps.size() - 1u);
    ++output_count;
    state->publications.push_back(PipelinePublish{
        .source = declared.source.buffer,
        .target = declared.target.buffer,
        .type = declared.source.type,
        .format = source_format,
        .source_offset = declared.source.offset,
        .target_offset = declared.target.offset,
        .count = declared.source.count,
        .target_stride = declared.target.stride,
        .element_bytes = declared.source.element_bytes,
        .window = state->steps[declared.step].window,
        .output = static_cast<std::uint16_t>(physical),
    });
    hash.number(source_ordinal);
    hash.number(*target);
    hash.number(static_cast<std::uint64_t>(declared.source.type));
    hash.format(source_format);
    hash.number(declared.source.count);
    hash.number(declared.target.offset);
    hash.number(declared.target.stride);
    hash.number(declared.source.element_bytes);
    hash.number(state->steps[declared.step].window);
    hash.number(physical);
  }
  state->transactional = build->commit;
  state->state_pairs.reserve(build->state_pairs.size());
  hash.number(build->state_pairs.size());
  for (const PipelineBuildStatePair &declared_pair : build->state_pairs) {
    const auto published = ordinals.find(declared_pair.published.buffer.get());
    const auto pending = ordinals.find(declared_pair.pending.buffer.get());
    if (published == ordinals.end() || pending == ordinals.end() ||
        published->second == pending->second) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineResource &published_resource =
        state->resources[published->second];
    const PipelineResource &pending_resource =
        state->resources[pending->second];
    if (published_resource.type != declared_pair.published.type ||
        pending_resource.type != declared_pair.pending.type ||
        published_resource.type != pending_resource.type) {
      return Status::fail(Reason::BindingTypeMismatch);
    }
    if (published_resource.count != pending_resource.count ||
        published_resource.bytes != pending_resource.bytes) {
      return Status::fail(Reason::ShapeMismatch);
    }
    if (!typed_format_matches(published_resource.type,
                              declared_pair.published.format,
                              published_resource.format) ||
        !typed_format_matches(pending_resource.type,
                              declared_pair.pending.format,
                              pending_resource.format)) {
      return Status::fail(Reason::FixedFormatMismatch);
    }

    PipelineResourceAdmission &published_admission =
        resource_admissions[published->second];
    PipelineResourceAdmission &pending_admission =
        resource_admissions[pending->second];
    // Phase 1 recorded the first read and first complete overwrite once per
    // canonical resource.  The original step-ordered law is equivalent to:
    // published is never written, pending has a complete overwrite, and no
    // pending read occurs before or in that first overwrite step (inputs are
    // admitted before outputs inside a step).
    if (published_resource.output != PipelineResource::no_output ||
        pending_admission.first_full_output_step ==
            PipelineResourceAdmission::none ||
        pending_admission.first_input_step <=
            pending_admission.first_full_output_step ||
        published_admission.partner != PipelineResourceAdmission::none ||
        pending_admission.partner != PipelineResourceAdmission::none) {
      return Status::fail(Reason::PipelineInvalid);
    }
    published_admission.partner = pending->second;
    pending_admission.partner = published->second;
    state->state_pairs.push_back(PipelineStatePair{
        .first = declared_pair.published.buffer,
        .second = declared_pair.pending.buffer,
        .type = published_resource.type,
        .format = published_resource.format,
        .count = published_resource.count,
        .bytes = published_resource.bytes,
    });
    hash.number(published->second);
    hash.number(pending->second);
    hash.number(static_cast<std::uint64_t>(published_resource.type));
    hash.format(published_resource.format);
    hash.number(published_resource.count);
    hash.number(published_resource.bytes);
  }
  state->resources.shrink_to_fit();

  prepare.state = std::move(state);
  prepare.steps = std::move(plan_steps);
  prepare.admissions = std::move(resource_admissions);
  prepare.outputs = std::move(physical_outputs);
  prepare.hash = hash;
  prepare.output_count = output_count;
  prepare.status_count = status_entry_count;
  prepare.binding_count = observed_bindings;
  return Status::success();
}

} // namespace rund::compute::detail
