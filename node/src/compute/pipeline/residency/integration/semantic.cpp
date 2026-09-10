#include "local.hpp"

#include "../../../device/residency/pool.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <limits>

namespace rund::compute::detail {

[[nodiscard]] Status
bind_graph_semantic_residency(const PipelineMemoryPlan &plan,
                              PipelineState &state) noexcept {
  const PipelineResidencyPlan &semantic_plan = plan.residency;
  const std::size_t count = semantic_plan.semantic_port_count;
  const bool reduction =
      semantic_plan.semantic == PipelineResidencySemantic::GraphReduction;
  const bool pointwise =
      semantic_plan.semantic == PipelineResidencySemantic::GraphPointwise;
  if (semantic_plan.pages == nullptr || semantic_plan.pool == nullptr ||
      semantic_plan.stage != PipelineResidencyStage::Graph ||
      semantic_plan.graph_stage != 0u || (!reduction && !pointwise) ||
      count < 2u || count > semantic_plan.semantic_ports.size() ||
      state.residency != semantic_plan.pages ||
      semantic_plan.frame_count == 0u ||
      semantic_plan.frame_count > PipelineLeafCapacity ||
      semantic_plan.bank >= residency::Pool::BankCount ||
      semantic_plan.first_step > plan.step_resources.size() ||
      semantic_plan.frame_count >
          plan.step_resources.size() - semantic_plan.first_step) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::TiledGraphPlan &graph = semantic_plan.pages->tiled_graph();
  std::array<std::uint32_t, residency::TiledGraphResourceCapacity>
      pipeline_resources{};
  pipeline_resources.fill(std::numeric_limits<std::uint32_t>::max());
  std::array<PipelineResidencyPort, residency::TiledGraphResourceCapacity>
      sealed{};
  std::uint32_t first_input = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t output_resource = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t control_resource = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t index = 0u; index < count; ++index) {
    const PipelineResidencySemanticPort &semantic_port =
        semantic_plan.semantic_ports[index];
    const bool read = semantic_port.access == residency::Access::Read;
    const bool write = semantic_port.access == residency::Access::Write;
    const bool expected_read = index + 1u < count;
    const residency::TiledGraphResource *const resource =
        graph.resource(semantic_port.resource);
    const residency::PoolPhysicalOwner *const owner =
        resource == nullptr
            ? nullptr
            : semantic_plan.pool->graph_owner(resource->physical_id);
    if (resource == nullptr || owner == nullptr ||
        owner->buffers[semantic_plan.bank] == nullptr || (!read && !write) ||
        read != expected_read ||
        semantic_port.program_port != (read ? index : 0u) ||
        (read &&
         (resource->kind != residency::GraphResourceKind::ExternalInput ||
          resource->role != residency::GraphResourceRole::Input ||
          resource->persistence != residency::ResourcePersistence::Backing)) ||
        (write &&
         (reduction
              ? (resource->kind != residency::GraphResourceKind::Internal ||
                 resource->role != residency::GraphResourceRole::Intermediate ||
                 resource->persistence !=
                     residency::ResourcePersistence::Transient)
              : (pointwise
                     ? (resource->kind !=
                            residency::GraphResourceKind::ExternalOutput ||
                        resource->role !=
                            residency::GraphResourceRole::Output ||
                        resource->persistence !=
                            residency::ResourcePersistence::Backing)
                     : true))) ||
        std::find_if(semantic_plan.semantic_ports.begin(),
                     semantic_plan.semantic_ports.begin() + index,
                     [resource](const PipelineResidencySemanticPort &prior) {
                       return prior.resource == resource->resource;
                     }) != semantic_plan.semantic_ports.begin() + index) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (read && first_input == std::numeric_limits<std::uint32_t>::max()) {
      first_input = semantic_port.resource;
    }
    if (write) {
      output_resource = semantic_port.resource;
    }
    residency::FrameRegion region{};
    if (!graph_frame_region(*semantic_plan.pool, semantic_plan.bank,
                            resource->physical_id, region)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    sealed[index] = PipelineResidencyPort{
        .resource = semantic_port.resource,
        .pipeline_resource = std::numeric_limits<std::uint32_t>::max(),
        .program_port = semantic_port.program_port,
        .access = semantic_port.access,
        .role = resource->role,
        .region = region,
        .page_bytes = resource->page_bytes,
    };
  }
  if (first_input == std::numeric_limits<std::uint32_t>::max() ||
      output_resource == std::numeric_limits<std::uint32_t>::max()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t input_count = count - 1u;
  for (std::size_t frame = 0u; frame < semantic_plan.frame_count; ++frame) {
    const PipelineStepResourcePlan &step =
        plan.step_resources[semantic_plan.first_step + frame];
    if (step.inputs.size() != input_count + 1u || step.outputs.size() != 1u ||
        step.physical_sources.size() != 1u || step.physical_sources[0] != 0u) {
      return Status::fail(Reason::PipelineInvalid);
    }
    for (std::size_t index = 0u; index < count; ++index) {
      const PipelineResidencySemanticPort &semantic_port =
          semantic_plan.semantic_ports[index];
      const bool read = semantic_port.access == residency::Access::Read;
      const residency::TiledGraphResource *const resource =
          graph.resource(semantic_port.resource);
      const residency::PoolPhysicalOwner *const owner =
          resource == nullptr
              ? nullptr
              : semantic_plan.pool->graph_owner(resource->physical_id);
      const PipelineResolvedViewPlan *const view =
          read ? &step.inputs[semantic_port.program_port]
               : &step.outputs[0u].view;
      std::uint64_t expected_offset = 0u;
      if (resource == nullptr || owner == nullptr ||
          !kernel::checked::mul(frame, resource->page_bytes, expected_offset) ||
          view->declared_type != resource->type ||
          view->declared_format != resource->format ||
          view->declared_access !=
              (read ? ResourceAccess::Read : ResourceAccess::Write) ||
          view->offset_bytes != expected_offset ||
          view->payload_bytes != resource->page_bytes ||
          view->resource >= state.resources.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (pipeline_resources[index] ==
          std::numeric_limits<std::uint32_t>::max()) {
        pipeline_resources[index] = view->resource;
      } else if (pipeline_resources[index] != view->resource) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineResource &physical = state.resources[view->resource];
      if (!physical.owned ||
          physical.buffer != owner->buffers[semantic_plan.bank] ||
          physical.type != resource->type ||
          physical.format != resource->format ||
          physical.bytes < resource->page_bytes ||
          (read != (physical.output == PipelineResource::no_output))) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
    const PipelineResolvedViewPlan &control = step.inputs[input_count];
    std::uint64_t expected_control_offset = 0u;
    if (!kernel::checked::mul(frame,
                              semantic_plan.pool->layout.control_page_bytes,
                              expected_control_offset) ||
        control.offset_bytes != expected_control_offset ||
        control.payload_bytes !=
            semantic_plan.pool->layout.control_page_bytes ||
        control.resource >= state.resources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (frame == 0u) {
      control_resource = control.resource;
    } else if (control_resource != control.resource) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  if (control_resource >= state.resources.size() ||
      state.resources[control_resource].buffer !=
          semantic_plan.pool->control[semantic_plan.bank] ||
      state.resources[control_resource].output != PipelineResource::no_output) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state.residency_input = pipeline_resources[0u];
  state.residency_output = pipeline_resources[count - 1u];
  state.residency_control = control_resource;
  state.residency_input_page_bytes =
      graph.resource(semantic_plan.semantic_ports[0u].resource)->page_bytes;
  state.residency_output_page_bytes =
      graph.resource(semantic_plan.semantic_ports[count - 1u].resource)
          ->page_bytes;
  state.residency_graph_stage = semantic_plan.graph_stage;
  state.residency_stage = semantic_plan.stage;
  state.residency_semantic = semantic_plan.semantic;
  state.residency_semantic_ports = semantic_plan.semantic_ports;
  state.residency_semantic_port_count = count;
  state.residency_ports = {};
  for (std::size_t index = 0u; index < count; ++index) {
    sealed[index].pipeline_resource = pipeline_resources[index];
  }
  for (std::size_t index = 0u; index < count; ++index) {
    state.residency_ports[index] = sealed[index];
  }
  state.residency_port_count = count;
  return state.residency_input < state.resources.size() &&
                 state.residency_output < state.resources.size()
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail
