#include "local.hpp"

#include "../../../device/residency/pool.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace rund::compute::detail {

[[nodiscard]] Status bind_graph_residency(const PipelineMemoryPlan &plan,
                                          PipelineState &state) noexcept {
  const PipelineResidencyPlan &residency_plan = plan.residency;
  const std::size_t first = residency_plan.first_step;
  const std::size_t frames = residency_plan.frame_count;
  const std::uint64_t input_page_bytes = residency_plan.input_page_bytes;
  const std::uint64_t output_page_bytes = residency_plan.output_page_bytes;
  const residency::TiledGraphPlan *const graph_plan =
      &residency_plan.pages->tiled_graph();
  const residency::TiledGraphStage *const graph_stage =
      residency_plan.graph_stage < graph_plan->stages().size()
          ? &graph_plan->stages()[residency_plan.graph_stage]
          : nullptr;
  if (state.residency != residency_plan.pages || frames == 0u ||
      frames > PipelineLeafCapacity || residency_plan.pool == nullptr ||
      residency_plan.bank >= residency::Pool::BankCount ||
      graph_stage == nullptr || graph_stage->ports.size() < 2u ||
      graph_stage->ports.size() > residency::TiledGraphPortCapacity ||
      (graph_stage->domain != residency::StageDomain::Tile &&
       graph_stage->domain != residency::StageDomain::TilePartial) ||
      first > plan.step_resources.size() ||
      frames > plan.step_resources.size() - first) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::size_t read_count = 0u;
  std::size_t write_count = 0u;
  for (const residency::TiledGraphPort port : graph_stage->ports) {
    read_count +=
        static_cast<std::size_t>(port.access == residency::Access::Read);
    write_count +=
        static_cast<std::size_t>(port.access == residency::Access::Write);
  }
  if (read_count == 0u || write_count == 0u ||
      graph_stage->active_count_input != read_count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<std::uint32_t, residency::TiledGraphPortCapacity> port_resources{};
  port_resources.fill(std::numeric_limits<std::uint32_t>::max());
  std::uint32_t control_resource = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t frame = 0u; frame < frames; ++frame) {
    const PipelineStepResourcePlan &step = plan.step_resources[first + frame];
    if (step.inputs.size() != read_count + 1u ||
        step.outputs.size() != write_count ||
        step.physical_sources.size() != write_count) {
      return Status::fail(Reason::PipelineInvalid);
    }
    for (std::size_t index = 0u; index < graph_stage->ports.size(); ++index) {
      const residency::TiledGraphPort port = graph_stage->ports[index];
      const residency::TiledGraphResource *const resource =
          graph_plan->resource(port.resource);
      const residency::PoolPhysicalOwner *const owner =
          resource == nullptr
              ? nullptr
              : residency_plan.pool->graph_owner(resource->physical_id);
      const bool read = port.access == residency::Access::Read;
      const PipelineResolvedViewPlan *const view =
          read ? (port.program_port < step.inputs.size()
                      ? &step.inputs[port.program_port]
                      : nullptr)
               : (port.program_port < step.outputs.size()
                      ? &step.outputs[port.program_port].view
                      : nullptr);
      std::uint64_t expected_offset = 0u;
      if (resource == nullptr || owner == nullptr || view == nullptr ||
          owner->buffers[residency_plan.bank] == nullptr ||
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
      if (frame == 0u) {
        port_resources[index] = view->resource;
      } else if (port_resources[index] != view->resource) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineResource &physical = state.resources[view->resource];
      const bool write = residency::writes(port.access);
      if (!physical.owned ||
          physical.buffer != owner->buffers[residency_plan.bank] ||
          physical.type != resource->type ||
          physical.format != resource->format ||
          physical.bytes < resource->page_bytes ||
          (write != (physical.output != PipelineResource::no_output))) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
    const PipelineResolvedViewPlan &control = step.inputs[read_count];
    std::uint64_t expected_control_offset = 0u;
    if (!kernel::checked::mul(frame,
                              residency_plan.pool->layout.control_page_bytes,
                              expected_control_offset) ||
        control.offset_bytes != expected_control_offset ||
        control.payload_bytes !=
            residency_plan.pool->layout.control_page_bytes ||
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
          residency_plan.pool->control[residency_plan.bank] ||
      state.resources[control_resource].output != PipelineResource::no_output) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state.residency_input = std::numeric_limits<std::uint32_t>::max();
  state.residency_output = std::numeric_limits<std::uint32_t>::max();
  state.residency_control = control_resource;
  state.residency_input_page_bytes = input_page_bytes;
  state.residency_output_page_bytes = output_page_bytes;
  state.residency_graph_stage = residency_plan.graph_stage;
  state.residency_semantic = PipelineResidencySemantic::None;
  state.residency_semantic_ports = {};
  state.residency_semantic_port_count = 0u;
  state.residency_port_count = graph_stage->ports.size();
  state.residency_ports = {};
  for (std::size_t index = 0u; index < graph_stage->ports.size(); ++index) {
    const residency::TiledGraphPort port = graph_stage->ports[index];
    const residency::TiledGraphResource *const resource =
        graph_plan->resource(port.resource);
    residency::FrameRegion region{};
    if (resource == nullptr || resource->remaps.size() > PipelineLeafCapacity ||
        !graph_frame_region(*residency_plan.pool, residency_plan.bank,
                            resource->physical_id, region)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    PipelineResidencyPort sealed_port{
        .resource = port.resource,
        .pipeline_resource = port_resources[index],
        .program_port = port.program_port,
        .access = port.access,
        .role = resource->role,
        .region = region,
        .page_bytes = resource->page_bytes,
        .remap_count = resource->remaps.size(),
    };
    std::copy(resource->remaps.begin(), resource->remaps.end(),
              sealed_port.remaps.begin());
    state.residency_ports[index] = std::move(sealed_port);
    if (port.access == residency::Access::Read &&
        state.residency_input == std::numeric_limits<std::uint32_t>::max()) {
      state.residency_input = port_resources[index];
    }
    if (port.access == residency::Access::Write &&
        state.residency_output == std::numeric_limits<std::uint32_t>::max()) {
      state.residency_output = port_resources[index];
    }
  }
  return state.residency_input < state.resources.size() &&
                 state.residency_output < state.resources.size()
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail
