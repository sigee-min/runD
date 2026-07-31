#include <rund/compute/pipeline.hpp>
#include <rund/compute/resource/plan.hpp>

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
  if (build->sealed || build->seed != nullptr) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (program == nullptr) {
    build->failure = Reason::ProgramInvalid;
    return;
  }
  if (build->logical_step_count >= PipelineStepCapacity ||
      inputs.size() > PipelineLeafCapacity ||
      outputs.size() > PipelineLeafCapacity - inputs.size() ||
      build->binding_count > PipelineBindingCapacity ||
      inputs.size() + outputs.size() >
          PipelineBindingCapacity - build->binding_count) {
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
                       const ResourceView *const resident,
                       const std::size_t maximum, const std::size_t tile,
                       const std::size_t terminal,
                       const std::uint32_t expected) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->seed != nullptr) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  if (program == nullptr) {
    build->failure = Reason::ProgramInvalid;
    return;
  }
  const std::size_t controls = resident == nullptr ? 0u : 2u;
  const std::size_t step_bindings = inputs.size() + controls + outputs.size();
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
      build->steps.size() > PipelineIterationCapacity - iterations ||
      !size::multiply(step_bindings, iterations, expanded_bindings) ||
      expanded_bindings > PipelineBindingCapacity - build->binding_count) {
    build->failure = Reason::PipelineCapacity;
    return;
  }
  const std::size_t first = build->steps.size();
  const std::size_t old_bindings = build->binding_count;
  const std::size_t old_internals = build->internals.size();
  const std::size_t old_publications = build->publications.size();
  try {
    std::vector<PipelineBinding> scratch;
    std::vector<PipelineBinding> alternate;
    scratch.reserve(outputs.size());
    alternate.reserve(resident == nullptr ? 0u : outputs.size());
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
    final.reserve(outputs.size());
    for (const ResourceView &output : outputs) {
      final.push_back(bind(output));
    }
    for (std::size_t iteration = 0u; iteration < iterations; ++iteration) {
      const bool final_bank =
          resident == nullptr && ((iterations - iteration) & 1u) != 0u;
      const std::span<const PipelineBinding> destination = [&] {
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
        binding.hidden = resident != nullptr || !final_bank;
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
  append_recurrence(build, program, inputs, outputs, iterations, nullptr, 0u,
                    0u, NoWindowTerminal, 1u);
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
  append_recurrence(build, program, inputs, outputs, iterations, &resident,
                    maximum, tile, terminal, expected);
}

void append_pipeline_state(const std::shared_ptr<PipelineBuildState> &build,
                           const std::shared_ptr<BufferState> &published,
                           const std::shared_ptr<BufferState> &pending,
                           const Type type, const FixedFormat format) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  if (build->sealed || build->seed != nullptr) {
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
      build->state_pairs.empty() || snapshot == nullptr ||
      build->seed != nullptr) {
    build->failure = Reason::PipelineInvalid;
    return;
  }
  build->seed = snapshot;
}

} // namespace rund::compute::detail
