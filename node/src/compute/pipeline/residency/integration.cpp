#include "integration.hpp"

#include "../../size.hpp"
#include "../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] PipelineBinding
slot_binding(const std::uint32_t owner, const Type type,
             const FixedFormat format, const std::size_t count,
             const std::size_t offset, const std::size_t backing_count,
             const ResourceAccess access) noexcept {
  const std::size_t width = type_bytes(type);
  const std::size_t bytes =
      width != 0u &&
              backing_count <= std::numeric_limits<std::size_t>::max() / width
          ? backing_count * width
          : 0u;
  return PipelineBinding{.type = type,
                         .format = format,
                         .offset = offset,
                         .count = count,
                         .stride = 1u,
                         .element_bytes = width,
                         .alignment = width,
                         .backing_bytes = bytes,
                         .access = access,
                         .owner = owner};
}

} // namespace

void append_pipeline_residency(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    std::shared_ptr<const residency::ResidencyPlan> pages,
    const std::uint64_t logical_bytes, const std::uint64_t input_page_bytes,
    const std::uint64_t output_page_bytes, const std::uint64_t staging_bytes,
    const std::uint64_t resident_bytes,
    const std::uint32_t slot_count) noexcept {
  const std::size_t first_step = build == nullptr ? 0u : build->steps.size();
  const std::size_t slot_bindings = static_cast<std::size_t>(slot_count) * 2u;
  std::size_t input_count = 0u;
  std::size_t output_count = 0u;
  if (build == nullptr || build->failure != Reason::Ok || build->sealed ||
      program == nullptr || program->device != build->device ||
      program->input_types.size() != 1u || program->input_sizes.size() != 1u ||
      program->input_formats.size() != 1u ||
      program->output_types.size() != 1u ||
      program->output_sizes.size() != 1u ||
      program->output_formats.size() != 1u || pages == nullptr ||
      !pages->identity() || logical_bytes == 0u || input_page_bytes == 0u ||
      output_page_bytes == 0u || staging_bytes == 0u || resident_bytes == 0u ||
      slot_count == 0u || slot_count > PipelineLeafCapacity ||
      first_step > PipelineStepCapacity ||
      slot_count > PipelineStepCapacity - first_step ||
      build->binding_count > PipelineBindingCapacity ||
      slot_bindings > PipelineBindingCapacity - build->binding_count ||
      build->internals.size() > PipelineResourceCapacity ||
      build->internals.size() > PipelineResourceCapacity - 2u ||
      !size::multiply(program->input_sizes[0], slot_count, input_count) ||
      !size::multiply(program->output_sizes[0], slot_count, output_count) ||
      first_step > std::numeric_limits<std::uint32_t>::max() ||
      build->residency.pages != nullptr) {
    if (build != nullptr && build->failure == Reason::Ok) {
      build->failure = Reason::PipelineInvalid;
    }
    return;
  }

  try {
    const std::uint32_t input_owner =
        static_cast<std::uint32_t>(build->internals.size());
    build->internals.push_back(PipelineInternal{
        .type = program->input_types[0],
        .format = program->input_formats[0],
        .count = input_count,
    });
    const std::uint32_t output_owner =
        static_cast<std::uint32_t>(build->internals.size());
    build->internals.push_back(PipelineInternal{
        .type = program->output_types[0],
        .format = program->output_formats[0],
        .count = output_count,
    });
    for (std::uint32_t slot = 0u; slot < slot_count; ++slot) {
      PipelineBuildStep step{};
      step.program = program;
      step.logical_step =
          static_cast<std::uint32_t>(build->logical_step_count++);
      step.inputs.push_back(slot_binding(
          input_owner, program->input_types[0], program->input_formats[0],
          program->input_sizes[0], slot * program->input_sizes[0], input_count,
          ResourceAccess::Read));
      step.outputs.push_back(slot_binding(
          output_owner, program->output_types[0], program->output_formats[0],
          program->output_sizes[0], slot * program->output_sizes[0],
          output_count, ResourceAccess::Write));
      build->steps.push_back(std::move(step));
    }
    build->binding_count += slot_bindings;
    build->residency = PipelineResidencyPlan{
        .pages = std::move(pages),
        .logical_bytes = logical_bytes,
        .input_page_bytes = input_page_bytes,
        .output_page_bytes = output_page_bytes,
        .staging_bytes = staging_bytes,
        .resident_bytes = resident_bytes,
        .first_step = static_cast<std::uint32_t>(first_step),
        .slot_count = slot_count,
    };
    build->memory.reset();
  } catch (const std::bad_alloc &) {
    build->failure = Reason::PipelineCapacity;
  }
}

Status bind_pipeline_residency(const PipelineMemoryPlan &plan,
                               PipelineState &state) noexcept {
  if (plan.residency.pages == nullptr) {
    return state.residency == nullptr ? Status::success()
                                      : Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t first = plan.residency.first_step;
  const std::size_t slots = plan.residency.slot_count;
  const std::uint64_t pair_bytes = plan.residency.pages->page_bytes();
  const std::uint64_t input_page_bytes = plan.residency.input_page_bytes;
  const std::uint64_t output_page_bytes = plan.residency.output_page_bytes;
  std::uint64_t expected_pair_bytes = 0u;
  if (state.residency != plan.residency.pages || slots == 0u ||
      slots > PipelineLeafCapacity || input_page_bytes == 0u ||
      output_page_bytes == 0u ||
      !kernel::checked::add(input_page_bytes, output_page_bytes,
                            expected_pair_bytes) ||
      pair_bytes != expected_pair_bytes || first > plan.step_resources.size() ||
      slots > plan.step_resources.size() - first) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::uint32_t input_resource = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t output_resource = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t slot = 0u; slot < slots; ++slot) {
    const PipelineStepResourcePlan &step = plan.step_resources[first + slot];
    std::uint64_t expected_input_offset = 0u;
    std::uint64_t expected_output_offset = 0u;
    if (step.inputs.size() != 1u || step.outputs.size() != 1u ||
        step.physical_sources.size() != 1u ||
        step.physical_sources[0] >= step.outputs.size() ||
        !kernel::checked::mul(slot, input_page_bytes, expected_input_offset) ||
        !kernel::checked::mul(slot, output_page_bytes,
                              expected_output_offset)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineResolvedViewPlan &input = step.inputs[0];
    const PipelineResolvedViewPlan &output =
        step.outputs[step.physical_sources[0]].view;
    if (slot == 0u) {
      input_resource = input.resource;
      output_resource = output.resource;
    }
    if (input.resource != input_resource ||
        output.resource != output_resource ||
        input.offset_bytes != expected_input_offset ||
        output.offset_bytes != expected_output_offset ||
        input.payload_bytes != input_page_bytes ||
        output.payload_bytes != output_page_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  std::uint64_t input_arena_bytes = 0u;
  std::uint64_t output_arena_bytes = 0u;
  if (!kernel::checked::mul(slots, input_page_bytes, input_arena_bytes) ||
      !kernel::checked::mul(slots, output_page_bytes, output_arena_bytes) ||
      input_resource >= state.resources.size() ||
      output_resource >= state.resources.size() ||
      input_resource == output_resource ||
      !state.resources[input_resource].owned ||
      !state.resources[output_resource].owned ||
      state.resources[input_resource].bytes != input_arena_bytes ||
      state.resources[output_resource].bytes != output_arena_bytes ||
      state.resources[output_resource].output == PipelineResource::no_output) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state.residency_input = input_resource;
  state.residency_output = output_resource;
  state.residency_input_page_bytes = input_page_bytes;
  state.residency_output_page_bytes = output_page_bytes;
  return Status::success();
}

} // namespace rund::compute::detail
