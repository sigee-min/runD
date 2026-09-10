#include "local.hpp"

#include "../../../../device/residency/pool.hpp"
#include "../../../../type.hpp"

#include <rund/compute/pipeline/shape.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail {

void append_pipeline_graph_semantic_residency(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    std::shared_ptr<const residency::ResidencyPlan> pages,
    std::shared_ptr<residency::Pool> pool, const std::uint64_t logical_bytes,
    const std::uint64_t resident_bytes, const std::uint32_t frame_count,
    const std::uint32_t bank, const std::uint32_t graph_stage,
    const PipelineResidencySemantic semantic,
    const std::span<const std::uint32_t> input_resources,
    const std::uint32_t output_resource) noexcept {
  if (build == nullptr || program == nullptr || pages == nullptr ||
      pool == nullptr || semantic == PipelineResidencySemantic::None ||
      graph_stage != 0u || bank >= residency::Pool::BankCount ||
      frame_count == 0u || frame_count > PipelineLeafCapacity ||
      input_resources.empty() ||
      input_resources.size() + 1u > residency::TiledGraphResourceCapacity ||
      output_resource == 0u ||
      program->input_types.size() != input_resources.size() + 1u ||
      program->input_formats.size() != program->input_types.size() ||
      program->input_sizes.size() != program->input_types.size() ||
      program->output_types.size() != 1u ||
      program->output_formats.size() != 1u ||
      program->output_sizes.size() != 1u) {
    if (build != nullptr) {
      build->failure = Reason::PipelineInvalid;
    }
    return;
  }
  const residency::TiledGraphPlan &graph = pages->tiled_graph();
  if (graph_stage >= graph.stages().size() ||
      graph.stages()[graph_stage].domain == residency::StageDomain::Terminal ||
      pool->control[bank] == nullptr ||
      program->input_types.back() != pool->layout.control_type ||
      program->input_formats.back() != pool->layout.control_format ||
      program->input_sizes.back() != 1u) {
    build->failure = Reason::PipelineInvalid;
    return;
  }

  std::array<const residency::TiledGraphResource *,
             residency::TiledGraphResourceCapacity>
      resources{};
  std::array<const residency::PoolPhysicalOwner *,
             residency::TiledGraphResourceCapacity>
      owners{};
  std::array<std::uint32_t, residency::TiledGraphResourceCapacity>
      pipeline_resources{};
  pipeline_resources.fill(std::numeric_limits<std::uint32_t>::max());
  for (std::size_t index = 0u; index < input_resources.size(); ++index) {
    const residency::TiledGraphResource *const resource =
        graph.resource(input_resources[index]);
    const residency::PoolPhysicalOwner *const owner =
        resource == nullptr ? nullptr
                            : pool->graph_owner(resource->physical_id);
    if (resource == nullptr || owner == nullptr ||
        resource->kind != residency::GraphResourceKind::ExternalInput ||
        resource->role != residency::GraphResourceRole::Input ||
        resource->persistence != residency::ResourcePersistence::Backing ||
        owner->buffers[bank] == nullptr || resource->page_bytes == 0u ||
        program->input_types[index] != resource->type ||
        program->input_formats[index] != resource->format ||
        program->input_sizes[index] * type_bytes(resource->type) !=
            resource->page_bytes ||
        std::find(input_resources.begin(), input_resources.begin() + index,
                  input_resources[index]) != input_resources.begin() + index ||
        std::find_if(owners.begin(), owners.begin() + index,
                     [owner](const auto *value) { return value == owner; }) !=
            owners.begin() + index) {
      build->failure = Reason::PipelineInvalid;
      return;
    }
    resources[index] = resource;
    owners[index] = owner;
  }
  const residency::TiledGraphResource *const output =
      graph.resource(output_resource);
  const residency::PoolPhysicalOwner *const output_owner =
      output == nullptr ? nullptr : pool->graph_owner(output->physical_id);
  const bool reduction = semantic == PipelineResidencySemantic::GraphReduction;
  if (output == nullptr || output_owner == nullptr ||
      output_owner->buffers[bank] == nullptr || output->page_bytes == 0u ||
      (reduction
           ? (output->role != residency::GraphResourceRole::Intermediate ||
              output->kind != residency::GraphResourceKind::Internal ||
              output->persistence != residency::ResourcePersistence::Transient)
           : (output->role != residency::GraphResourceRole::Output ||
              output->kind != residency::GraphResourceKind::ExternalOutput ||
              output->persistence !=
                  residency::ResourcePersistence::Backing)) ||
      program->output_types[0] != output->type ||
      program->output_formats[0] != output->format ||
      program->output_sizes[0] * type_bytes(output->type) !=
          output->page_bytes ||
      std::find(input_resources.begin(), input_resources.end(),
                output_resource) != input_resources.end()) {
    build->failure = Reason::PipelineInvalid;
    return;
  }

  const std::size_t input_count = input_resources.size();
  const std::size_t output_index = input_count + 1u;
  const std::size_t first_step = build->steps.size();
  const std::size_t frame_bindings = frame_count * (input_count + 2u);
  try {
    build->internals.reserve(build->internals.size() + output_index + 1u);
    for (std::size_t index = 0u; index < input_count; ++index) {
      build->internals.push_back(PipelineInternal{
          .owner = owners[index]->buffers[bank],
          .type = resources[index]->type,
          .format = resources[index]->format,
          .count = owners[index]->buffers[bank]->count,
      });
    }
    build->internals.push_back(PipelineInternal{
        .owner = pool->control[bank],
        .type = pool->layout.control_type,
        .format = pool->layout.control_format,
        .count = pool->control[bank]->count,
    });
    build->internals.push_back(PipelineInternal{
        .owner = output_owner->buffers[bank],
        .type = output->type,
        .format = output->format,
        .count = output_owner->buffers[bank]->count,
    });
    for (std::uint32_t frame = 0u; frame < frame_count; ++frame) {
      PipelineBuildStep step{};
      step.program = program;
      step.logical_step =
          static_cast<std::uint32_t>(build->logical_step_count++);
      step.inputs.reserve(input_count + 1u);
      step.outputs.reserve(1u);
      for (std::size_t index = 0u; index < input_count; ++index) {
        step.inputs.push_back(append_frame_binding(
            static_cast<std::uint32_t>(build->internals.size() -
                                       (input_count + 2u) + index),
            program->input_types[index], program->input_formats[index],
            program->input_sizes[index],
            static_cast<std::size_t>(frame) * program->input_sizes[index],
            owners[index]->buffers[bank]->count, ResourceAccess::Read));
      }
      step.inputs.push_back(append_frame_binding(
          static_cast<std::uint32_t>(build->internals.size() - 2u),
          program->input_types.back(), program->input_formats.back(), 1u, frame,
          pool->control[bank]->count, ResourceAccess::Read));
      step.outputs.push_back(append_frame_binding(
          static_cast<std::uint32_t>(build->internals.size() - 1u),
          program->output_types[0], program->output_formats[0],
          program->output_sizes[0],
          static_cast<std::size_t>(frame) * program->output_sizes[0],
          output_owner->buffers[bank]->count, ResourceAccess::Write));
      build->steps.push_back(std::move(step));
    }
    if (build->logical_step_count != build->steps.size() ||
        build->steps.size() != frame_count) {
      build->failure = Reason::PipelineInvalid;
      return;
    }
    PipelineResidencyPlan residency{
        .pages = std::move(pages),
        .pool = std::move(pool),
        .logical_bytes = logical_bytes,
        .input_page_bytes = resources[0]->page_bytes,
        .output_page_bytes = output->page_bytes,
        .resident_bytes = resident_bytes,
        .first_step = static_cast<std::uint32_t>(first_step),
        .frame_count = frame_count,
        .bank = bank,
        .graph_stage = graph_stage,
        .stage = PipelineResidencyStage::Graph,
        .semantic = semantic,
        .semantic_port_count = input_count + 1u,
    };
    for (std::size_t index = 0u; index < input_count; ++index) {
      residency.semantic_ports[index] = PipelineResidencySemanticPort{
          .resource = input_resources[index],
          .program_port = static_cast<std::uint16_t>(index),
          .access = residency::Access::Read,
      };
    }
    residency.semantic_ports[input_count] = PipelineResidencySemanticPort{
        .resource = output_resource,
        .program_port = 0u,
        .access = residency::Access::Write,
    };
    build->binding_count += frame_bindings;
    build->residency = std::move(residency);
    build->memory.reset();
  } catch (const std::bad_alloc &) {
    build->failure = Reason::PipelineCapacity;
  }
}

} // namespace rund::compute::detail
