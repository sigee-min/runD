#include "local.hpp"

#include "../../../../device/residency/pool.hpp"
#include "../../../../size.hpp"
#include "../../../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail {

void append_pipeline_residency(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    std::shared_ptr<const residency::ResidencyPlan> pages,
    std::shared_ptr<residency::Pool> pool, const std::uint64_t logical_bytes,
    const std::uint64_t input_page_bytes, const std::uint64_t output_page_bytes,
    const std::uint64_t resident_bytes, const std::uint32_t frame_count,
    const std::uint32_t bank) noexcept {
  const std::size_t first_step = build == nullptr ? 0u : build->steps.size();
  const std::size_t frame_bindings = static_cast<std::size_t>(frame_count) * 2u;
  std::size_t input_count = 0u;
  std::size_t output_count = 0u;
  if (build == nullptr || build->failure != Reason::Ok || build->sealed ||
      program == nullptr || program->device != build->device ||
      program->input_types.size() != 1u || program->input_sizes.size() != 1u ||
      program->input_formats.size() != 1u ||
      program->output_types.size() != 1u ||
      program->output_sizes.size() != 1u ||
      program->output_formats.size() != 1u || pages == nullptr ||
      pool == nullptr || !pages->identity() || logical_bytes == 0u ||
      input_page_bytes == 0u || output_page_bytes == 0u ||
      resident_bytes == 0u || frame_count == 0u ||
      bank >= residency::Pool::BankCount || pool->input[bank] == nullptr ||
      pool->output[bank] == nullptr || frame_count > PipelineLeafCapacity ||
      first_step > PipelineStepCapacity ||
      frame_count > PipelineStepCapacity - first_step ||
      build->binding_count > PipelineBindingCapacity ||
      frame_bindings > PipelineBindingCapacity - build->binding_count ||
      build->internals.size() > PipelineResourceCapacity ||
      build->internals.size() > PipelineResourceCapacity - 2u ||
      !size::multiply(program->input_sizes[0], frame_count, input_count) ||
      !size::multiply(program->output_sizes[0], frame_count, output_count) ||
      first_step > std::numeric_limits<std::uint32_t>::max() ||
      build->residency.pages != nullptr) {
    if (build != nullptr && build->failure == Reason::Ok) {
      build->failure = Reason::PipelineInvalid;
    }
    return;
  }
  // A compatible Pool may borrow a K-prefix from a canonical C-frame input
  // arena. The internal resource must retain the complete physical owner;
  // prepared bindings below still address exactly the first K page slots.
  input_count = pool->input[bank]->count;

  try {
    const std::uint32_t input_owner =
        static_cast<std::uint32_t>(build->internals.size());
    build->internals.push_back(PipelineInternal{
        .owner = pool->input[bank],
        .type = program->input_types[0],
        .format = program->input_formats[0],
        .count = input_count,
    });
    const std::uint32_t output_owner =
        static_cast<std::uint32_t>(build->internals.size());
    build->internals.push_back(PipelineInternal{
        .owner = pool->output[bank],
        .type = program->output_types[0],
        .format = program->output_formats[0],
        .count = output_count,
    });
    for (std::uint32_t frame = 0u; frame < frame_count; ++frame) {
      PipelineBuildStep step{};
      step.program = program;
      step.logical_step =
          static_cast<std::uint32_t>(build->logical_step_count++);
      step.inputs.push_back(append_frame_binding(
          input_owner, program->input_types[0], program->input_formats[0],
          program->input_sizes[0], frame * program->input_sizes[0], input_count,
          ResourceAccess::Read));
      step.outputs.push_back(append_frame_binding(
          output_owner, program->output_types[0], program->output_formats[0],
          program->output_sizes[0], frame * program->output_sizes[0],
          output_count, ResourceAccess::Write));
      build->steps.push_back(std::move(step));
    }
    build->binding_count += frame_bindings;
    build->residency = PipelineResidencyPlan{
        .pages = std::move(pages),
        .pool = std::move(pool),
        .logical_bytes = logical_bytes,
        .input_page_bytes = input_page_bytes,
        .output_page_bytes = output_page_bytes,
        .resident_bytes = resident_bytes,
        .first_step = static_cast<std::uint32_t>(first_step),
        .frame_count = frame_count,
        .bank = bank,
    };
    build->memory.reset();
  } catch (const std::bad_alloc &) {
    build->failure = Reason::PipelineCapacity;
  }
}

} // namespace rund::compute::detail
