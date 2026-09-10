#include "local.hpp"

#include "../../../../device/residency/pool.hpp"
#include "../../../../size.hpp"
#include "../../../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail {

void append_pipeline_graph_residency(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    std::shared_ptr<const residency::ResidencyPlan> pages,
    std::shared_ptr<residency::Pool> pool, const std::uint64_t logical_bytes,
    const std::uint64_t resident_bytes, const std::uint32_t frame_count,
    const std::uint32_t bank, const std::uint32_t graph_stage) noexcept {
  const std::size_t first_step = build == nullptr ? 0u : build->steps.size();
  const residency::TiledGraphPlan *const graph_plan =
      pages != nullptr && pages->graph_tiled() ? &pages->tiled_graph()
                                               : nullptr;
  const residency::TiledGraphStage *const sealed_stage =
      graph_plan != nullptr && graph_stage < graph_plan->stages().size()
          ? &graph_plan->stages()[graph_stage]
          : nullptr;
  const std::shared_ptr<BufferState> control_owner =
      pool == nullptr || bank >= residency::Pool::BankCount
          ? nullptr
          : pool->control[bank];
  const std::size_t port_count =
      sealed_stage == nullptr ? 0u : sealed_stage->ports.size();
  const std::size_t frame_bindings =
      static_cast<std::size_t>(frame_count) * (port_count + 1u);
  std::array<const residency::TiledGraphResource *,
             residency::TiledGraphPortCapacity>
      sealed_resources{};
  std::array<const residency::PoolPhysicalOwner *,
             residency::TiledGraphPortCapacity>
      physical_owners{};
  std::array<std::uint32_t, residency::TiledGraphPortCapacity>
      pipeline_resources{};
  std::size_t read_count = 0u;
  std::size_t write_count = 0u;
  std::size_t control_count = 0u;
  if (build == nullptr || build->failure != Reason::Ok || build->sealed ||
      program == nullptr || program->device != build->device ||
      pages == nullptr || !pages->graph_tiled() || pool == nullptr ||
      !pages->identity() || logical_bytes == 0u || resident_bytes == 0u ||
      frame_count == 0u || bank >= residency::Pool::BankCount ||
      sealed_stage == nullptr ||
      (sealed_stage->domain != residency::StageDomain::Tile &&
       sealed_stage->domain != residency::StageDomain::TilePartial) ||
      port_count < 2u || port_count > residency::TiledGraphPortCapacity ||
      control_owner == nullptr || pool->layout.control_page_bytes == 0u ||
      frame_count > PipelineLeafCapacity || first_step > PipelineStepCapacity ||
      frame_count > PipelineStepCapacity - first_step ||
      build->binding_count > PipelineBindingCapacity ||
      frame_bindings > PipelineBindingCapacity - build->binding_count ||
      build->internals.size() > PipelineResourceCapacity ||
      port_count + 1u > PipelineResourceCapacity - build->internals.size() ||
      first_step > std::numeric_limits<std::uint32_t>::max() ||
      build->residency.pages != nullptr) {
    if (build != nullptr && build->failure == Reason::Ok) {
      build->failure = Reason::PipelineInvalid;
    }
    return;
  }
  for (std::size_t index = 0u; index < port_count; ++index) {
    const residency::TiledGraphPort &port = sealed_stage->ports[index];
    const residency::TiledGraphResource *const resource =
        graph_plan->resource(port.resource);
    const residency::PoolPhysicalOwner *const owner =
        resource == nullptr ? nullptr
                            : pool->graph_owner(resource->physical_id);
    const bool read = port.access == residency::Access::Read;
    const std::size_t ordinal = port.program_port;
    const bool valid_program_port =
        read ? ordinal < program->input_types.size() &&
                   ordinal < program->input_sizes.size() &&
                   ordinal < program->input_formats.size()
             : port.access == residency::Access::Write &&
                   ordinal < program->output_types.size() &&
                   ordinal < program->output_sizes.size() &&
                   ordinal < program->output_formats.size();
    std::size_t program_bytes = 0u;
    if (resource == nullptr || owner == nullptr ||
        owner->buffers[bank] == nullptr || !valid_program_port ||
        !(read ? size::multiply(program->input_sizes[ordinal],
                                type_bytes(program->input_types[ordinal]),
                                program_bytes)
               : size::multiply(program->output_sizes[ordinal],
                                type_bytes(program->output_types[ordinal]),
                                program_bytes)) ||
        resource->type != (read ? program->input_types[ordinal]
                                : program->output_types[ordinal]) ||
        resource->format != (read ? program->input_formats[ordinal]
                                  : program->output_formats[ordinal]) ||
        resource->page_bytes != program_bytes) {
      build->failure = Reason::PipelineInvalid;
      return;
    }
    sealed_resources[index] = resource;
    physical_owners[index] = owner;
    read_count += static_cast<std::size_t>(read);
    write_count += static_cast<std::size_t>(!read);
  }
  if (read_count == 0u || write_count == 0u ||
      sealed_stage->active_count_input != read_count ||
      program->input_types.size() != read_count + 1u ||
      program->input_sizes.size() != read_count + 1u ||
      program->input_formats.size() != read_count + 1u ||
      program->output_types.size() != write_count ||
      program->output_sizes.size() != write_count ||
      program->output_formats.size() != write_count ||
      program->input_types[read_count] != pool->layout.control_type ||
      program->input_formats[read_count] != pool->layout.control_format ||
      program->input_sizes[read_count] != 1u ||
      !size::multiply(frame_count, program->input_sizes[read_count],
                      control_count)) {
    build->failure = Reason::PipelineInvalid;
    return;
  }

  try {
    for (std::size_t index = 0u; index < port_count; ++index) {
      pipeline_resources[index] =
          static_cast<std::uint32_t>(build->internals.size());
      build->internals.push_back(PipelineInternal{
          .owner = physical_owners[index]->buffers[bank],
          .type = sealed_resources[index]->type,
          .format = sealed_resources[index]->format,
          .count = physical_owners[index]->buffers[bank]->count,
      });
    }
    const std::uint32_t control_resource =
        static_cast<std::uint32_t>(build->internals.size());
    build->internals.push_back(PipelineInternal{
        .owner = control_owner,
        .type = program->input_types[read_count],
        .format = program->input_formats[read_count],
        .count = control_count,
    });
    for (std::uint32_t frame = 0u; frame < frame_count; ++frame) {
      PipelineBuildStep step{};
      step.program = program;
      step.logical_step =
          static_cast<std::uint32_t>(build->logical_step_count++);
      for (std::size_t ordinal = 0u; ordinal < read_count; ++ordinal) {
        const auto found =
            std::find_if(sealed_stage->ports.begin(), sealed_stage->ports.end(),
                         [ordinal](const residency::TiledGraphPort &port) {
                           return port.access == residency::Access::Read &&
                                  port.program_port == ordinal;
                         });
        const std::size_t index =
            static_cast<std::size_t>(found - sealed_stage->ports.begin());
        const std::size_t backing_count =
            physical_owners[index]->buffers[bank]->count;
        step.inputs.push_back(append_frame_binding(
            pipeline_resources[index], program->input_types[ordinal],
            program->input_formats[ordinal], program->input_sizes[ordinal],
            frame * program->input_sizes[ordinal], backing_count,
            ResourceAccess::Read));
      }
      step.inputs.push_back(append_frame_binding(
          control_resource, program->input_types[read_count],
          program->input_formats[read_count], 1u, frame, control_count,
          ResourceAccess::Read));
      for (std::size_t ordinal = 0u; ordinal < write_count; ++ordinal) {
        const auto found =
            std::find_if(sealed_stage->ports.begin(), sealed_stage->ports.end(),
                         [ordinal](const residency::TiledGraphPort &port) {
                           return port.access == residency::Access::Write &&
                                  port.program_port == ordinal;
                         });
        const std::size_t index =
            static_cast<std::size_t>(found - sealed_stage->ports.begin());
        const std::size_t backing_count =
            physical_owners[index]->buffers[bank]->count;
        step.outputs.push_back(append_frame_binding(
            pipeline_resources[index], program->output_types[ordinal],
            program->output_formats[ordinal], program->output_sizes[ordinal],
            frame * program->output_sizes[ordinal], backing_count,
            ResourceAccess::Write));
      }
      build->steps.push_back(std::move(step));
    }
    build->binding_count += frame_bindings;
    build->residency = PipelineResidencyPlan{
        .pages = std::move(pages),
        .pool = std::move(pool),
        .logical_bytes = logical_bytes,
        .input_page_bytes = sealed_resources[0]->page_bytes,
        .output_page_bytes =
            sealed_resources[static_cast<std::size_t>(
                                 std::find_if(
                                     sealed_stage->ports.begin(),
                                     sealed_stage->ports.end(),
                                     [](const residency::TiledGraphPort &port) {
                                       return port.access ==
                                              residency::Access::Write;
                                     }) -
                                 sealed_stage->ports.begin())]
                ->page_bytes,
        .resident_bytes = resident_bytes,
        .first_step = static_cast<std::uint32_t>(first_step),
        .frame_count = frame_count,
        .bank = bank,
        .graph_stage = graph_stage,
        .stage = PipelineResidencyStage::Graph,
    };
    build->memory.reset();
  } catch (const std::bad_alloc &) {
    build->failure = Reason::PipelineCapacity;
  }
}

} // namespace rund::compute::detail
