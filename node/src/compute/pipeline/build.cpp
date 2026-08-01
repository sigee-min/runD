#include <rund/compute/pipeline.hpp>
#include <rund/compute/resource/plan.hpp>

#include "../program/output.hpp"
#include "../size.hpp"
#include "../type.hpp"
#include "state.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <new>

namespace rund::compute::detail {
namespace {

[[nodiscard]] PipelineBinding bind(const ResourceView &view,
                                   const bool hidden = false) noexcept {
  return PipelineBinding{.buffer = view.buffer,
                         .type = view.type,
                         .format = view.format,
                         .offset = view.offset,
                         .count = view.count,
                         .stride = view.stride,
                         .element_bytes = view.element_bytes,
                         .alignment = view.alignment,
                         .backing_bytes =
                             view.buffer == nullptr ? 0u : view.buffer->bytes,
                         .access = view.access,
                         .hidden = hidden};
}

[[nodiscard]] PipelineBinding
bind(const std::uint32_t owner, const PipelineInternal &resource,
     const ResourceAccess access, const std::size_t offset = 0u,
     const std::size_t count = 0u, const bool hidden = false) noexcept {
  const std::size_t elements = count == 0u ? resource.count : count;
  const std::size_t width = type_bytes(resource.type);
  const std::size_t backing =
      width != 0u &&
              resource.count <= std::numeric_limits<std::size_t>::max() / width
          ? resource.count * width
          : 0u;
  return PipelineBinding{.type = resource.type,
                         .format = resource.format,
                         .offset = offset,
                         .count = elements,
                         .stride = 1u,
                         .element_bytes = width,
                         .alignment = width,
                         .backing_bytes = backing,
                         .access = access,
                         .owner = owner,
                         .hidden = hidden};
}

[[nodiscard]] Result<bool> intersects(const PipelineBinding &left,
                                      const ResourceView &right) noexcept {
  if (left.buffer == nullptr || right.buffer == nullptr ||
      left.buffer != right.buffer || left.element_bytes == 0u ||
      right.element_bytes == 0u) {
    return Result<bool>::success(false);
  }
  const auto scaled = [](const std::size_t value,
                         const std::size_t width) noexcept {
    return value > std::numeric_limits<std::uint64_t>::max() / width
               ? std::numeric_limits<std::uint64_t>::max()
               : static_cast<std::uint64_t>(value) * width;
  };
  const resource::Resource owner{
      .id = 1u,
      .bytes = left.buffer->bytes,
      .alias_group = 1u,
  };
  const resource::Access a{
      .resource = 1u,
      .offset_bytes = scaled(left.offset, left.element_bytes),
      .element_bytes = left.element_bytes,
      .element_count = left.count,
      .stride_bytes = scaled(left.stride, left.element_bytes),
  };
  const resource::Access b{
      .resource = 1u,
      .offset_bytes = scaled(right.offset, right.element_bytes),
      .element_bytes = right.element_bytes,
      .element_count = right.count,
      .stride_bytes = scaled(right.stride, right.element_bytes),
  };
  return resource::intersects(owner, a, owner, b);
}

[[nodiscard]] Status route(const PipelineBuildState &build,
                           const ResourceView &view,
                           PipelineBinding &result) noexcept {
  result = bind(view);
  for (const PipelineBuildPublish &publication : build.publications) {
    const PipelineBinding &target = publication.target;
    if (target.buffer != view.buffer) {
      continue;
    }
    auto overlap = intersects(target, view);
    if (!overlap) {
      return Status::fail(overlap.reason());
    }
    if (!*overlap) {
      if (view.access == ResourceAccess::Write) {
        // Publication output identity is currently Buffer-owned.  A second
        // disjoint public write would otherwise be accepted here and rejected
        // later as a duplicate output owner.  Reject it at the authored edge
        // instead of allowing terminal publication to clobber or obscure it.
        return Status::fail(Reason::BindingAliasUnsupported);
      }
      continue;
    }
    if (publication.kind == PipelinePublishKind::Window) {
      if (view.access != ResourceAccess::Read) {
        // The append-only target has one writer for the complete Pipeline.
        return Status::fail(Reason::BindingAliasUnsupported);
      }
      // Declaration order has already sealed the complete nested window.
      // Unlike terminal recurrence, the O(Tile) private source cannot stand
      // in for the O(Max) result. Keep the caller target as the read binding;
      // resource analysis then owns the exact publication-to-read barrier.
      return Status::success();
    }
    const PipelineBinding &source = publication.source;
    if (target.type != view.type || target.format != view.format ||
        target.element_bytes != view.element_bytes || target.stride == 0u ||
        view.offset < target.offset) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
    const std::size_t delta = view.offset - target.offset;
    if (delta % target.stride != 0u || view.stride % target.stride != 0u) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
    const std::size_t first = delta / target.stride;
    const std::size_t step = view.stride / target.stride;
    std::size_t distance = 0u;
    std::size_t last = first;
    const bool tail_overflow =
        view.count != 0u && step != 0u &&
        (!size::multiply(view.count - 1u, step, distance) ||
         !size::add(first, distance, last));
    if (source.stride == 0u || step == 0u || first >= target.count ||
        tail_overflow || (view.count != 0u && last >= target.count)) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
    std::size_t source_delta = 0u;
    std::size_t source_offset = 0u;
    std::size_t source_stride = 0u;
    if (!size::multiply(first, source.stride, source_delta) ||
        !size::add(source.offset, source_delta, source_offset) ||
        !size::multiply(step, source.stride, source_stride)) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
    result = source;
    result.offset = source_offset;
    result.count = view.count;
    result.stride = source_stride;
    result.alignment = view.alignment;
    result.access = view.access;
    result.hidden = true;
    return Status::success();
  }
  return Status::success();
}

void changed(PipelineBuildState &build) noexcept { build.memory.reset(); }

[[nodiscard]] bool has_seed(const PipelineBuildState &build) noexcept {
  return build.seed != nullptr || build.storage_seed != nullptr ||
         build.device_seed != nullptr;
}

} // namespace

std::shared_ptr<PipelineBuildState>
make_pipeline(const std::shared_ptr<DeviceState> &device) noexcept {
  try {
    auto state = std::make_shared<PipelineBuildState>();
    state->device = device;
    state->steps.reserve(PipelineStepCapacity);
    state->state_pairs.reserve(PipelineLeafCapacity);
    state->publications.reserve(PipelineLeafCapacity);
    state->internals.reserve(PipelineLeafCapacity);
    state->nested_windows.reserve(PipelineStepCapacity);
    if (device == nullptr) {
      state->failure = Reason::DeviceInvalid;
    }
    return state;
  } catch (const std::bad_alloc &) {
    return {};
  }
}

void append_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                     const std::shared_ptr<ProgramState> &program,
                     const std::span<const ResourceView> inputs,
                     const std::span<const ResourceView> outputs) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (program == nullptr) {
    build->failure = Reason::ProgramInvalid;
    return;
  }
  const std::size_t binding_capacity = build->nested_windows.empty()
                                           ? PipelineBindingCapacity
                                           : PipelineRouteBindingCapacity;
  const std::size_t route_capacity = build->nested_windows.empty()
                                         ? PipelineIterationCapacity
                                         : PipelineRouteCapacity;
  if (build->logical_step_count >= PipelineStepCapacity ||
      build->steps.size() >= route_capacity ||
      inputs.size() > PipelineLeafCapacity ||
      outputs.size() > PipelineLeafCapacity - inputs.size() ||
      build->binding_count > binding_capacity ||
      inputs.size() + outputs.size() >
          binding_capacity - build->binding_count) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  try {
    PipelineBuildStep step{};
    step.logical_step = static_cast<std::uint32_t>(build->logical_step_count);
    step.program = program;
    step.inputs.reserve(inputs.size());
    step.outputs.reserve(outputs.size());
    for (const ResourceView &view : inputs) {
      if (view.access != ResourceAccess::Read) {
        build->failure = Reason::BindingInvalid;
        return;
      }
      PipelineBinding binding{};
      const Status routed = route(*build, view, binding);
      if (!routed) {
        build->failure = routed.reason();
        return;
      }
      step.inputs.push_back(std::move(binding));
    }
    for (const ResourceView &view : outputs) {
      if (view.access != ResourceAccess::Write) {
        build->failure = Reason::BindingInvalid;
        return;
      }
      PipelineBinding binding{};
      const Status routed = route(*build, view, binding);
      if (!routed) {
        build->failure = routed.reason();
        return;
      }
      step.outputs.push_back(std::move(binding));
    }
    build->binding_count += inputs.size() + outputs.size();
    build->steps.push_back(std::move(step));
    ++build->logical_step_count;
    changed(*build);
  } catch (const std::bad_alloc &) {
    build->failure = Reason::PipelineCapacity;
  }
}

namespace {

void append_recurrence(const std::shared_ptr<PipelineBuildState> &build,
                       const std::shared_ptr<ProgramState> &program,
                       const std::span<const ResourceView> inputs,
                       const std::span<const ResourceView> outputs,
                       const std::size_t iterations,
                       const bool write_each_iteration,
                       const ResourceView *const resident,
                       const std::size_t maximum, const std::size_t tile,
                       const std::size_t terminal,
                       const std::uint32_t expected) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (program == nullptr) {
    build->failure = Reason::ProgramInvalid;
    return;
  }
  const std::size_t controls = resident == nullptr ? 0u : 2u;
  const std::size_t step_bindings = inputs.size() + controls + outputs.size();
  const std::size_t route_capacity = build->nested_windows.empty()
                                         ? PipelineIterationCapacity
                                         : PipelineRouteCapacity;
  const std::size_t binding_capacity = build->nested_windows.empty()
                                           ? PipelineBindingCapacity
                                           : PipelineRouteBindingCapacity;
  std::size_t expanded_bindings = 0u;
  if (iterations == 0u || iterations > PipelineIterationCapacity ||
      outputs.empty() || outputs.size() > inputs.size() ||
      (terminal != NoWindowTerminal &&
       (terminal >= outputs.size() || outputs[terminal].type != Type::U32 ||
        inputs[terminal].type != Type::U32 || outputs[terminal].count != 1u ||
        inputs[terminal].count != 1u)) ||
      inputs.size() + controls > PipelineLeafCapacity ||
      outputs.size() > PipelineLeafCapacity - inputs.size() - controls ||
      build->logical_step_count >= PipelineStepCapacity ||
      build->steps.size() > route_capacity - iterations ||
      !size::multiply(step_bindings, iterations, expanded_bindings) ||
      build->binding_count > binding_capacity ||
      expanded_bindings > binding_capacity - build->binding_count) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  const bool write_each = write_each_iteration && iterations > 1u;
  const std::size_t first = build->steps.size();
  const std::size_t old_bindings = build->binding_count;
  const std::size_t old_internals = build->internals.size();
  const std::size_t old_publications = build->publications.size();
  try {
    std::vector<PipelineBinding> scratch;
    std::vector<PipelineBinding> alternate;
    std::vector<PipelineBinding> history;
    std::vector<std::size_t> history_counts;
    scratch.reserve(write_each ? 0u : outputs.size());
    alternate.reserve(resident == nullptr ? 0u : outputs.size());
    history.reserve(write_each ? outputs.size() : 0u);
    history_counts.reserve(write_each ? outputs.size() : 0u);
    for (const ResourceView &output : outputs) {
      if (output.access != ResourceAccess::Write) {
        build->failure = Reason::BindingInvalid;
        return;
      }
      const std::size_t bytes = type_bytes(output.type);
      if (bytes == 0u || !size::multiply(output.count, bytes) ||
          output.element_bytes != bytes) {
        build->failure = Reason::ShapeMismatch;
        return;
      }
      if (write_each) {
        if (output.count % iterations != 0u) {
          build->failure = Reason::ShapeMismatch;
          return;
        }
        history.push_back(bind(output));
        history_counts.push_back(output.count / iterations);
        continue;
      }
      const auto owner = static_cast<std::uint32_t>(build->internals.size());
      build->internals.push_back(PipelineInternal{
          .type = output.type, .format = output.format, .count = output.count});
      scratch.push_back(bind(owner, build->internals.back(),
                             ResourceAccess::Write, 0u, output.count, true));
      if (resident != nullptr) {
        const auto second = static_cast<std::uint32_t>(build->internals.size());
        build->internals.push_back(PipelineInternal{.type = output.type,
                                                    .format = output.format,
                                                    .count = output.count});
        alternate.push_back(bind(second, build->internals.back(),
                                 ResourceAccess::Write, 0u, output.count,
                                 true));
      }
    }
    std::vector<PipelineBinding> current;
    current.reserve(inputs.size());
    if (std::any_of(inputs.begin(), inputs.end(), [](const ResourceView &view) {
          return view.access != ResourceAccess::Read;
        })) {
      build->failure = Reason::BindingInvalid;
      return;
    }
    for (const ResourceView &input : inputs) {
      PipelineBinding binding{};
      const Status routed = route(*build, input, binding);
      if (!routed) {
        build->failure = routed.reason();
        return;
      }
      current.push_back(std::move(binding));
    }

    std::uint32_t ordinal_owner = PipelineBinding::external;
    if (resident != nullptr) {
      if (resident->access != ResourceAccess::Read ||
          resident->type != Type::U32 || resident->count != 1u ||
          maximum == 0u || tile == 0u || tile > maximum) {
        build->failure = Reason::BindingInvalid;
        return;
      }
      ordinal_owner = static_cast<std::uint32_t>(build->internals.size());
      build->internals.push_back(PipelineInternal{
          .type = Type::U32,
          .count = iterations,
          .fill = PipelineFill::Ordinal,
      });
    }

    std::vector<PipelineBinding> final;
    final.reserve(write_each ? 0u : outputs.size());
    if (!write_each) {
      for (const ResourceView &output : outputs) {
        final.push_back(bind(output));
      }
    }
    std::vector<PipelineBinding> each;
    each.reserve(write_each ? outputs.size() : 0u);
    for (std::size_t iteration = 0u; iteration < iterations; ++iteration) {
      each.clear();
      if (write_each) {
        for (std::size_t index = 0u; index < history.size(); ++index) {
          PipelineBinding binding = history[index];
          std::size_t first_element = 0u;
          std::size_t offset = 0u;
          if (!size::multiply(iteration, history_counts[index],
                              first_element) ||
              !size::multiply(first_element, binding.stride, first_element) ||
              !size::add(binding.offset, first_element, offset)) {
            build->failure = Reason::ShapeMismatch;
            return;
          }
          binding.offset = offset;
          binding.count = history_counts[index];
          each.push_back(std::move(binding));
        }
      }
      const bool final_bank =
          resident == nullptr && ((iterations - iteration) & 1u) != 0u;
      const std::span<const PipelineBinding> destination = [&] {
        if (write_each) {
          return std::span<const PipelineBinding>{each};
        }
        if (final_bank) {
          return std::span<const PipelineBinding>{final};
        }
        return resident != nullptr && (iteration & 1u) != 0u
                   ? std::span<const PipelineBinding>{alternate}
                   : std::span<const PipelineBinding>{scratch};
      }();
      PipelineBuildStep step{};
      step.program = program;
      step.logical_step = static_cast<std::uint32_t>(build->logical_step_count);
      step.iteration = static_cast<std::uint32_t>(iteration);
      step.iteration_bound = static_cast<std::uint32_t>(iterations);
      step.window_max = maximum;
      step.window_tile = tile;
      step.window_terminal = terminal;
      step.window_expected = expected;
      step.writes_each_iteration = write_each;
      step.inputs.reserve(inputs.size() + controls);
      step.outputs.reserve(outputs.size());
      for (PipelineBinding binding : current) {
        binding.access = ResourceAccess::Read;
        step.inputs.push_back(std::move(binding));
      }
      if (resident != nullptr) {
        step.inputs.push_back(bind(*resident));
        step.inputs.push_back(bind(ordinal_owner,
                                   build->internals[ordinal_owner],
                                   ResourceAccess::Read, iteration, 1u));
      }
      for (PipelineBinding binding : destination) {
        binding.access = ResourceAccess::Write;
        binding.hidden = !write_each && (resident != nullptr || !final_bank);
        step.outputs.push_back(std::move(binding));
      }
      build->steps.push_back(std::move(step));
      for (std::size_t index = 0u; index < outputs.size(); ++index) {
        current[index] = destination[index];
        current[index].access = ResourceAccess::Read;
      }
    }
    if (resident != nullptr) {
      for (std::size_t index = 0u; index < outputs.size(); ++index) {
        build->publications.push_back(PipelineBuildPublish{
            .source = current[index],
            .target = final[index],
            .count = {},
            .step = first,
            .output = static_cast<std::uint32_t>(index),
        });
      }
    }
    build->binding_count += expanded_bindings;
    ++build->logical_step_count;
    changed(*build);
  } catch (const std::bad_alloc &) {
    build->steps.resize(first);
    build->internals.resize(old_internals);
    build->publications.resize(old_publications);
    build->binding_count = old_bindings;
    build->failure = Reason::PipelineCapacity;
  }
}

} // namespace

void append_pipeline_repeat(const std::shared_ptr<PipelineBuildState> &build,
                            const std::shared_ptr<ProgramState> &program,
                            const std::span<const ResourceView> inputs,
                            const std::span<const ResourceView> outputs,
                            const std::size_t iterations) noexcept {
  append_recurrence(build, program, inputs, outputs, iterations, false, nullptr,
                    0u, 0u, NoWindowTerminal, 1u);
}

void append_pipeline_repeat_each(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    const std::span<const ResourceView> inputs,
    const std::span<const ResourceView> outputs,
    const std::size_t iterations) noexcept {
  append_recurrence(build, program, inputs, outputs, iterations, true, nullptr,
                    0u, 0u, NoWindowTerminal, 1u);
}

void append_pipeline_windows(const std::shared_ptr<PipelineBuildState> &build,
                             const std::shared_ptr<ProgramState> &program,
                             const ResourceView &resident,
                             const std::span<const ResourceView> inputs,
                             const std::span<const ResourceView> outputs,
                             const std::size_t maximum, const std::size_t tile,
                             const std::size_t terminal,
                             const std::uint32_t expected) noexcept {
  std::size_t rounded = 0u;
  if (tile == 0u || maximum == 0u || tile > maximum ||
      maximum > std::numeric_limits<std::uint32_t>::max() ||
      !size::add(maximum, tile - 1u, rounded)) {
    if (build != nullptr && build->failure == Reason::Ok) {
      build->failure = Reason::PipelineCapacity;
    }
    return;
  }
  const std::size_t iterations = rounded / tile;
  append_recurrence(build, program, inputs, outputs, iterations, false,
                    &resident, maximum, tile, terminal, expected);
}

void append_pipeline_window_repeat(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &seed,
    const std::shared_ptr<ProgramState> &action,
    const std::shared_ptr<ProgramState> &fold, const ResourceView &resident,
    const std::span<const ResourceView> inputs,
    const std::span<const ResourceView> final_outputs,
    const std::span<const ResourceView> window_outputs,
    const std::size_t maximum, const std::size_t tile, const std::size_t inner,
    const std::size_t terminal, const std::uint32_t expected) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (seed == nullptr || fold == nullptr ||
      ((inner == 0u) != (action == nullptr))) {
    build->failure = Reason::ProgramInvalid;
    return;
  }

  std::size_t rounded = 0u;
  if (maximum == 0u || tile == 0u || tile > maximum ||
      maximum > std::numeric_limits<std::uint32_t>::max() ||
      inner > PipelineInnerIterationCapacity ||
      !size::add(maximum, tile - 1u, rounded)) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  const std::size_t outer = rounded / tile;
  const std::size_t seed_output_count =
      output_count(seed->output_aliases, seed->output_types.size());
  const std::size_t action_output_count =
      action == nullptr
          ? 0u
          : output_count(action->output_aliases, action->output_types.size());
  const std::size_t fold_output_count =
      output_count(fold->output_aliases, fold->output_types.size());
  const std::size_t recurrent_count = final_outputs.size();
  const std::size_t window_count = window_outputs.size();
  if (outer == 0u || outer > PipelineIterationCapacity ||
      seed_output_count == 0u || recurrent_count == 0u ||
      fold_output_count != recurrent_count + window_count ||
      inputs.size() < recurrent_count ||
      seed->input_types.size() != inputs.size() - recurrent_count + 2u ||
      (action != nullptr &&
       (action_output_count == 0u || action_output_count > seed_output_count ||
        seed_output_count != action->input_types.size())) ||
      fold->input_types.size() != recurrent_count + seed_output_count ||
      (terminal != NoWindowTerminal &&
       (terminal >= recurrent_count ||
        fold->output_types[terminal] != Type::U32 ||
        fold->output_sizes[terminal] != 1u)) ||
      build->logical_step_count >= PipelineStepCapacity) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  if (!seed->output_aliases.empty() ||
      (action != nullptr && !action->output_aliases.empty()) ||
      !fold->output_aliases.empty()) {
    build->failure = Reason::BindingAliasUnsupported;
    return;
  }
  const std::size_t route_count = outer + inner + 3u;
  if (route_count > PipelineRouteCapacity ||
      build->steps.size() > PipelineRouteCapacity - route_count ||
      seed->input_types.size() + seed_output_count > PipelineLeafCapacity ||
      (action != nullptr && action->input_types.size() + action_output_count >
                                PipelineLeafCapacity) ||
      fold->input_types.size() + fold_output_count > PipelineLeafCapacity) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  if (resident.access != ResourceAccess::Read || resident.type != Type::U32 ||
      resident.count != 1u || resident.element_bytes != sizeof(std::uint32_t) ||
      std::any_of(inputs.begin(), inputs.end(),
                  [](const ResourceView &view) {
                    return view.access != ResourceAccess::Read;
                  }) ||
      std::any_of(final_outputs.begin(), final_outputs.end(),
                  [](const ResourceView &view) {
                    return view.access != ResourceAccess::Write;
                  }) ||
      std::any_of(window_outputs.begin(), window_outputs.end(),
                  [](const ResourceView &view) {
                    return view.access != ResourceAccess::Write;
                  })) {
    build->failure = Reason::BindingInvalid;
    return;
  }

  const auto same_slot = [](const Type left_type, const FixedFormat left_format,
                            const std::size_t left_count, const Type right_type,
                            const FixedFormat right_format,
                            const std::size_t right_count) noexcept {
    return left_type == right_type && left_format == right_format &&
           left_count == right_count;
  };
  const std::size_t seed_external_count = inputs.size() - recurrent_count;
  for (std::size_t index = 0u; index < seed_external_count; ++index) {
    const ResourceView &view = inputs[recurrent_count + index];
    if (!same_slot(view.type, view.format, view.count, seed->input_types[index],
                   seed->input_formats[index], seed->input_sizes[index])) {
      build->failure = Reason::ShapeMismatch;
      return;
    }
  }
  const std::size_t count_input = seed->input_types.size() - 2u;
  if (seed->input_types[count_input] != Type::U32 ||
      seed->input_types[count_input + 1u] != Type::U32 ||
      seed->input_sizes[count_input] != 1u ||
      seed->input_sizes[count_input + 1u] != 1u) {
    build->failure = Reason::ShapeMismatch;
    return;
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
      build->failure = Reason::ShapeMismatch;
      return;
    }
  }
  for (std::size_t index = 0u; action != nullptr && index < action_output_count;
       ++index) {
    if (!same_slot(action->output_types[index], action->output_formats[index],
                   action->output_sizes[index], action->input_types[index],
                   action->input_formats[index], action->input_sizes[index])) {
      build->failure = Reason::ShapeMismatch;
      return;
    }
  }
  for (std::size_t index = 0u; index < recurrent_count; ++index) {
    if (!same_slot(inputs[index].type, inputs[index].format,
                   inputs[index].count, fold->input_types[index],
                   fold->input_formats[index], fold->input_sizes[index]) ||
        !same_slot(final_outputs[index].type, final_outputs[index].format,
                   final_outputs[index].count, fold->output_types[index],
                   fold->output_formats[index], fold->output_sizes[index]) ||
        !same_slot(fold->output_types[index], fold->output_formats[index],
                   fold->output_sizes[index], fold->input_types[index],
                   fold->input_formats[index], fold->input_sizes[index])) {
      build->failure = Reason::ShapeMismatch;
      return;
    }
  }
  for (std::size_t index = 0u; index < window_count; ++index) {
    const ResourceView &target = window_outputs[index];
    const std::size_t output = recurrent_count + index;
    if (!same_slot(target.type, target.format, tile, fold->output_types[output],
                   fold->output_formats[output], fold->output_sizes[output]) ||
        target.count != maximum || target.stride != 1u) {
      build->failure = Reason::ShapeMismatch;
      return;
    }
  }
  const auto overlaps = [](const ResourceView &left,
                           const ResourceView &right) {
    return intersects(bind(left), right);
  };
  for (std::size_t index = 0u; index < window_count; ++index) {
    const ResourceView &target = window_outputs[index];
    for (std::size_t other = 0u; other < index; ++other) {
      if (target.buffer == window_outputs[other].buffer) {
        build->failure = Reason::BindingDuplicate;
        return;
      }
      auto overlap = overlaps(target, window_outputs[other]);
      if (!overlap || *overlap) {
        build->failure =
            overlap ? Reason::BindingAliasUnsupported : overlap.reason();
        return;
      }
    }
    auto resident_overlap = overlaps(target, resident);
    if (!resident_overlap || *resident_overlap) {
      build->failure = resident_overlap ? Reason::BindingAliasUnsupported
                                        : resident_overlap.reason();
      return;
    }
    for (const ResourceView &input : inputs) {
      auto overlap = overlaps(target, input);
      if (!overlap || *overlap) {
        build->failure =
            overlap ? Reason::BindingAliasUnsupported : overlap.reason();
        return;
      }
    }
    for (const ResourceView &output : final_outputs) {
      auto overlap = overlaps(target, output);
      if (!overlap || *overlap) {
        build->failure =
            overlap ? Reason::BindingAliasUnsupported : overlap.reason();
        return;
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
      !size::multiply(seed_bindings, outer, seed_bindings) ||
      (action != nullptr && !size::add(action->input_types.size(),
                                       action_output_count, action_bindings)) ||
      !size::multiply(action_bindings, inner, action_bindings) ||
      !size::add(fold->input_types.size(), fold_output_count, fold_bindings) ||
      !size::multiply(fold_bindings, 3u, fold_bindings) ||
      !add_bindings(seed_bindings) || !add_bindings(action_bindings) ||
      !add_bindings(fold_bindings) ||
      build->binding_count > PipelineRouteBindingCapacity ||
      binding_count > PipelineRouteBindingCapacity - build->binding_count) {
    build->failure = Reason::PipelineCapacity;
    return;
  }

  const std::size_t old_steps = build->steps.size();
  const std::size_t old_bindings = build->binding_count;
  const std::size_t old_internals = build->internals.size();
  const std::size_t old_publications = build->publications.size();
  const std::size_t old_nested = build->nested_windows.size();
  try {
    const auto internal = [&](const Type type, const FixedFormat format,
                              const std::size_t count,
                              const ResourceAccess access) {
      const auto owner = static_cast<std::uint32_t>(build->internals.size());
      build->internals.push_back(
          PipelineInternal{.type = type, .format = format, .count = count});
      return bind(owner, build->internals.back(), access, 0u, count, true);
    };
    std::vector<PipelineBinding> outer_seed;
    std::vector<PipelineBinding> seed_external;
    std::vector<PipelineBinding> outer_first;
    std::vector<PipelineBinding> outer_second;
    std::vector<PipelineBinding> final;
    std::vector<PipelineBinding> window_tile;
    std::vector<PipelineBinding> window_target;
    outer_seed.reserve(recurrent_count);
    seed_external.reserve(seed_external_count);
    outer_first.reserve(recurrent_count);
    outer_second.reserve(recurrent_count);
    final.reserve(recurrent_count);
    window_tile.reserve(window_count);
    window_target.reserve(window_count);
    for (std::size_t index = 0u; index < recurrent_count; ++index) {
      PipelineBinding routed{};
      const Status status = route(*build, inputs[index], routed);
      if (!status) {
        build->failure = status.reason();
        throw build->failure;
      }
      outer_seed.push_back(std::move(routed));
      outer_first.push_back(
          internal(fold->output_types[index], fold->output_formats[index],
                   fold->output_sizes[index], ResourceAccess::Write));
      outer_second.push_back(
          internal(fold->output_types[index], fold->output_formats[index],
                   fold->output_sizes[index], ResourceAccess::Write));
      final.push_back(bind(final_outputs[index]));
    }
    for (std::size_t index = 0u; index < window_count; ++index) {
      const std::size_t output = recurrent_count + index;
      window_tile.push_back(internal(fold->output_types[output],
                                     fold->output_formats[output], tile,
                                     ResourceAccess::Write));
      window_target.push_back(bind(window_outputs[index]));
    }
    for (std::size_t index = 0u; index < seed_external_count; ++index) {
      PipelineBinding routed{};
      const Status status =
          route(*build, inputs[recurrent_count + index], routed);
      if (!status) {
        build->failure = status.reason();
        throw build->failure;
      }
      seed_external.push_back(std::move(routed));
    }

    std::vector<PipelineBinding> tile_first;
    std::vector<PipelineBinding> tile_second;
    tile_first.reserve(seed_output_count);
    tile_second.reserve(action_output_count);
    for (std::size_t index = 0u; index < seed_output_count; ++index) {
      tile_first.push_back(
          internal(seed->output_types[index], seed->output_formats[index],
                   seed->output_sizes[index], ResourceAccess::Write));
      if (index < action_output_count) {
        tile_second.push_back(
            internal(action->output_types[index], action->output_formats[index],
                     action->output_sizes[index], ResourceAccess::Write));
      }
    }

    const auto ordinal_owner =
        static_cast<std::uint32_t>(build->internals.size());
    build->internals.push_back(PipelineInternal{
        .type = Type::U32, .count = outer, .fill = PipelineFill::Ordinal});

    const auto nested =
        static_cast<std::uint16_t>(build->nested_windows.size() + 1u);
    if (nested == 0u) {
      build->failure = Reason::PipelineCapacity;
      throw build->failure;
    }
    const std::size_t seed_first = build->steps.size();
    for (std::size_t ordinal = 0u; ordinal < outer; ++ordinal) {
      PipelineBuildStep step{};
      step.program = seed;
      step.logical_step = static_cast<std::uint32_t>(build->logical_step_count);
      step.iteration = static_cast<std::uint32_t>(ordinal);
      step.iteration_bound = static_cast<std::uint32_t>(outer);
      step.nested = nested;
      step.route = PipelineRoute::NestedSeed;
      step.inputs.reserve(seed->input_types.size());
      step.outputs.reserve(seed_output_count);
      for (PipelineBinding binding : seed_external) {
        binding.access = ResourceAccess::Read;
        step.inputs.push_back(std::move(binding));
      }
      step.inputs.push_back(bind(resident));
      step.inputs.push_back(bind(ordinal_owner, build->internals[ordinal_owner],
                                 ResourceAccess::Read, ordinal, 1u, true));
      for (PipelineBinding binding : tile_first) {
        binding.access = ResourceAccess::Write;
        binding.hidden = true;
        step.outputs.push_back(std::move(binding));
      }
      build->steps.push_back(std::move(step));
    }

    const std::size_t action_first = build->steps.size();
    for (std::size_t iteration = 0u; iteration < inner; ++iteration) {
      const bool even = (iteration & 1u) == 0u;
      PipelineBuildStep step{};
      step.program = action;
      step.logical_step = static_cast<std::uint32_t>(build->logical_step_count);
      step.iteration = static_cast<std::uint32_t>(iteration);
      step.iteration_bound = static_cast<std::uint32_t>(inner);
      step.nested = nested;
      step.route = PipelineRoute::NestedAction;
      step.inputs.reserve(seed_output_count);
      step.outputs.reserve(action_output_count);
      for (std::size_t index = 0u; index < seed_output_count; ++index) {
        PipelineBinding binding = index < action_output_count && !even
                                      ? tile_second[index]
                                      : tile_first[index];
        binding.access = ResourceAccess::Read;
        step.inputs.push_back(std::move(binding));
      }
      for (std::size_t index = 0u; index < action_output_count; ++index) {
        PipelineBinding binding = even ? tile_second[index] : tile_first[index];
        binding.access = ResourceAccess::Write;
        binding.hidden = true;
        step.outputs.push_back(std::move(binding));
      }
      build->steps.push_back(std::move(step));
    }

    std::vector<PipelineBinding> tile_final = tile_first;
    if ((inner & 1u) != 0u) {
      std::copy(tile_second.begin(), tile_second.end(), tile_final.begin());
    }
    const std::size_t fold_first = build->steps.size();
    for (std::size_t route_index = 0u; route_index < 3u; ++route_index) {
      const std::span<const PipelineBinding> current =
          route_index == 0u
              ? std::span<const PipelineBinding>{outer_seed}
              : (route_index == 1u
                     ? std::span<const PipelineBinding>{outer_first}
                     : std::span<const PipelineBinding>{outer_second});
      const std::span<const PipelineBinding> destination =
          route_index == 1u ? std::span<const PipelineBinding>{outer_second}
                            : std::span<const PipelineBinding>{outer_first};
      PipelineBuildStep step{};
      step.program = fold;
      step.logical_step = static_cast<std::uint32_t>(build->logical_step_count);
      step.iteration = static_cast<std::uint32_t>(route_index);
      step.iteration_bound = 3u;
      step.nested = nested;
      step.route = PipelineRoute::NestedFold;
      step.inputs.reserve(fold->input_types.size());
      step.outputs.reserve(fold_output_count);
      for (PipelineBinding binding : current) {
        binding.access = ResourceAccess::Read;
        step.inputs.push_back(std::move(binding));
      }
      for (PipelineBinding binding : tile_final) {
        binding.access = ResourceAccess::Read;
        step.inputs.push_back(std::move(binding));
      }
      for (PipelineBinding binding : destination) {
        binding.access = ResourceAccess::Write;
        binding.hidden = true;
        step.outputs.push_back(std::move(binding));
      }
      for (PipelineBinding binding : window_tile) {
        binding.access = ResourceAccess::Write;
        binding.hidden = true;
        step.outputs.push_back(std::move(binding));
      }
      build->steps.push_back(std::move(step));
    }

    const std::span<const PipelineBinding> final_bank =
        (outer & 1u) != 0u ? std::span<const PipelineBinding>{outer_first}
                           : std::span<const PipelineBinding>{outer_second};
    for (std::size_t index = 0u; index < recurrent_count; ++index) {
      build->publications.push_back(PipelineBuildPublish{
          .source = final_bank[index],
          .target = final[index],
          .count = {},
          .step = fold_first,
          .output = static_cast<std::uint32_t>(index),
      });
    }
    for (std::size_t index = 0u; index < window_count; ++index) {
      build->publications.push_back(PipelineBuildPublish{
          .source = window_tile[index],
          .target = window_target[index],
          .count = bind(resident),
          .step = fold_first,
          .output = static_cast<std::uint32_t>(recurrent_count + index),
          .maximum = maximum,
          .tile = tile,
          .kind = PipelinePublishKind::Window,
      });
    }
    build->nested_windows.push_back(PipelineBuildNestedWindow{
        .count = bind(resident),
        .begin = old_steps,
        .end = build->steps.size(),
        .seed_first = seed_first,
        .seed_count = outer,
        .action_first = action_first,
        .action_count = inner,
        .fold_first = fold_first,
        .recurrent_output_count = recurrent_count,
        .maximum = maximum,
        .tile = tile,
        .terminal = terminal,
        .expected = expected,
    });
    build->binding_count += binding_count;
    ++build->logical_step_count;
    changed(*build);
  } catch (const Reason) {
    build->steps.resize(old_steps);
    build->internals.resize(old_internals);
    build->publications.resize(old_publications);
    build->nested_windows.resize(old_nested);
    build->binding_count = old_bindings;
  } catch (const std::bad_alloc &) {
    build->steps.resize(old_steps);
    build->internals.resize(old_internals);
    build->publications.resize(old_publications);
    build->nested_windows.resize(old_nested);
    build->binding_count = old_bindings;
    build->failure = Reason::PipelineCapacity;
  }
}

void append_pipeline_state(const std::shared_ptr<PipelineBuildState> &build,
                           const std::shared_ptr<BufferState> &published,
                           const std::shared_ptr<BufferState> &pending,
                           const Type type, const FixedFormat format) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (published == nullptr || pending == nullptr || published == pending) {
    build->failure = published == pending ? Reason::BindingDuplicate
                                          : Reason::BindingInvalid;
    return;
  }
  if (published->device != build->device || pending->device != build->device) {
    build->failure = Reason::BindingDeviceMismatch;
    return;
  }
  if (!valid_type(type) || published->type != type || pending->type != type) {
    build->failure = Reason::BindingTypeMismatch;
    return;
  }
  const std::size_t element_bytes = type_bytes(type);
  std::size_t expected_bytes = 0u;
  if (element_bytes == 0u || published->count != pending->count ||
      published->bytes != pending->bytes ||
      !size::multiply(published->count, element_bytes, expected_bytes) ||
      published->bytes != expected_bytes ||
      published->physical_bytes < published->bytes ||
      pending->physical_bytes < pending->bytes) {
    build->failure = Reason::ShapeMismatch;
    return;
  }
  for (const PipelineBuildStatePair &pair : build->state_pairs) {
    if (pair.published.buffer == published ||
        pair.published.buffer == pending || pair.pending.buffer == published ||
        pair.pending.buffer == pending) {
      build->failure = Reason::BindingDuplicate;
      return;
    }
  }
  if (build->state_pairs.size() >= PipelineLeafCapacity) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  try {
    const auto binding = [&](const std::shared_ptr<BufferState> &buffer,
                             const ResourceAccess access) {
      return PipelineBinding{.buffer = buffer,
                             .type = type,
                             .format = format,
                             .count = buffer->count,
                             .stride = 1u,
                             .element_bytes = element_bytes,
                             .alignment = element_bytes,
                             .backing_bytes = buffer->bytes,
                             .access = access};
    };
    build->state_pairs.push_back(PipelineBuildStatePair{
        .published = binding(published, ResourceAccess::Read),
        .pending = binding(pending, ResourceAccess::Write),
    });
    changed(*build);
  } catch (const std::bad_alloc &) {
    build->failure = Reason::PipelineCapacity;
  }
}

void configure_pipeline_profile(
    const std::shared_ptr<PipelineBuildState> &build,
    const PipelineProfile profile) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (profile != PipelineProfile::None && profile != PipelineProfile::Steps) {
    build->failure = Reason::ProfileInvalid;
    return;
  }
  build->profile = profile;
  changed(*build);
}

void configure_pipeline_sealed_repetitions(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::size_t repetitions) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || has_seed(*build) ||
      build->sealed_repetitions_configured || repetitions == 0u ||
      repetitions > PipelineSealedRepetitionCapacity ||
      repetitions > std::numeric_limits<std::uint32_t>::max()) {
    build->failure =
        repetitions == 0u || repetitions > PipelineSealedRepetitionCapacity
            ? Reason::PipelineCapacity
            : Reason::PipelineInvalid;
    return;
  }
  build->sealed_repetitions = static_cast<std::uint32_t>(repetitions);
  build->sealed_repetitions_configured = true;
  changed(*build);
}

void commit_pipeline(
    const std::shared_ptr<PipelineBuildState> &build) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->state_pairs.empty()) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  build->commit = true;
  build->sealed = true;
}

void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->commit || build->steps.empty() ||
      build->state_pairs.empty() || snapshot == nullptr || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  build->seed = snapshot;
}

void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->commit || build->steps.empty() ||
      build->state_pairs.empty() || publication == nullptr ||
      has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  build->device_seed = publication;
}

void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->commit || build->steps.empty() ||
      build->state_pairs.empty() || storage == nullptr || has_seed(*build)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  build->storage_seed = storage;
}

} // namespace rund::compute::detail
