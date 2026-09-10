#include "local.hpp"

#include "../../../device/residency/pool.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::compute::detail {

[[nodiscard]] Status bind_direct_residency(const PipelineMemoryPlan &plan,
                                           PipelineState &state) noexcept {
  const std::size_t first = plan.residency.first_step;
  const std::size_t frames = plan.residency.frame_count;
  const std::uint64_t planned_page_bytes = plan.residency.pages->page_bytes();
  const std::uint64_t input_page_bytes = plan.residency.input_page_bytes;
  const std::uint64_t output_page_bytes = plan.residency.output_page_bytes;
  const bool direct = plan.residency.stage == PipelineResidencyStage::Direct;
  const bool graph = plan.residency.stage == PipelineResidencyStage::Graph;
  const residency::TiledGraphPlan *const graph_plan =
      graph ? &plan.residency.pages->tiled_graph() : nullptr;
  const residency::TiledGraphStage *const graph_stage =
      graph_plan != nullptr &&
              plan.residency.graph_stage < graph_plan->stages().size()
          ? &graph_plan->stages()[plan.residency.graph_stage]
          : nullptr;
  std::uint64_t expected_page_bytes = 0u;
  if (state.residency != plan.residency.pages || frames == 0u ||
      frames > PipelineLeafCapacity || input_page_bytes == 0u ||
      output_page_bytes == 0u || plan.residency.pool == nullptr ||
      plan.residency.bank >= residency::Pool::BankCount ||
      (direct != plan.residency.pages->streamed()) ||
      (graph != plan.residency.pages->graph_tiled()) ||
      (direct && plan.residency.graph_stage != residency::NoGraphStage) ||
      (graph && (graph_stage == nullptr ||
                 (graph_stage->domain != residency::StageDomain::Tile &&
                  graph_stage->domain != residency::StageDomain::TilePartial) ||
                 graph_stage->ports.size() != 2u)) ||
      (direct && !kernel::checked::add(input_page_bytes, output_page_bytes,
                                       expected_page_bytes)) ||
      (graph &&
       (!kernel::checked::add(
            plan.residency.pool->layout.input_page_bytes,
            plan.residency.pool->layout.intermediate_page_bytes,
            expected_page_bytes) ||
        !kernel::checked::add(expected_page_bytes,
                              plan.residency.pool->layout.output_page_bytes,
                              expected_page_bytes))) ||
      planned_page_bytes != expected_page_bytes ||
      first > plan.step_resources.size() ||
      frames > plan.step_resources.size() - first) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::uint32_t input_resource = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t control_resource = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t output_resource = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t frame = 0u; frame < frames; ++frame) {
    const PipelineStepResourcePlan &step = plan.step_resources[first + frame];
    std::uint64_t expected_input_offset = 0u;
    std::uint64_t expected_control_offset = 0u;
    std::uint64_t expected_output_offset = 0u;
    if (step.inputs.size() != (direct ? 1u : 2u) || step.outputs.size() != 1u ||
        step.physical_sources.size() != 1u ||
        step.physical_sources[0] >= step.outputs.size() ||
        !kernel::checked::mul(frame, input_page_bytes, expected_input_offset) ||
        (graph && !kernel::checked::mul(
                      frame, plan.residency.pool->layout.control_page_bytes,
                      expected_control_offset)) ||
        !kernel::checked::mul(frame, output_page_bytes,
                              expected_output_offset)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineResolvedViewPlan &input = step.inputs[0];
    const PipelineResolvedViewPlan &output =
        step.outputs[step.physical_sources[0]].view;
    if (frame == 0u) {
      input_resource = input.resource;
      if (graph) {
        control_resource = step.inputs[1].resource;
      }
      output_resource = output.resource;
    }
    if (input.resource != input_resource ||
        (graph && (step.inputs[1].resource != control_resource ||
                   step.inputs[1].offset_bytes != expected_control_offset ||
                   step.inputs[1].payload_bytes !=
                       plan.residency.pool->layout.control_page_bytes)) ||
        output.resource != output_resource ||
        input.offset_bytes != expected_input_offset ||
        output.offset_bytes != expected_output_offset ||
        input.payload_bytes != input_page_bytes ||
        output.payload_bytes != output_page_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  std::uint64_t input_arena_bytes = 0u;
  std::uint64_t control_arena_bytes = 0u;
  std::uint64_t output_arena_bytes = 0u;
  const residency::TiledGraphResource *const graph_input_resource =
      graph && graph_stage->ports.size() == 2u
          ? graph_plan->resource(graph_stage->ports[0].resource)
          : nullptr;
  const residency::TiledGraphResource *const graph_output_resource =
      graph && graph_stage->ports.size() == 2u
          ? graph_plan->resource(graph_stage->ports[1].resource)
          : nullptr;
  const residency::PoolPhysicalOwner *const graph_input_owner =
      graph_input_resource == nullptr
          ? nullptr
          : plan.residency.pool->graph_owner(graph_input_resource->physical_id);
  const residency::PoolPhysicalOwner *const graph_output_owner =
      graph_output_resource == nullptr
          ? nullptr
          : plan.residency.pool->graph_owner(
                graph_output_resource->physical_id);
  const std::shared_ptr<BufferState> expected_input =
      direct ? plan.residency.pool->input[plan.residency.bank]
      : graph_input_owner == nullptr
          ? nullptr
          : graph_input_owner->buffers[plan.residency.bank];
  const std::shared_ptr<BufferState> expected_output =
      direct ? plan.residency.pool->output[plan.residency.bank]
      : graph_output_owner == nullptr
          ? nullptr
          : graph_output_owner->buffers[plan.residency.bank];
  if (!kernel::checked::mul(frames, input_page_bytes, input_arena_bytes) ||
      (graph && !kernel::checked::mul(
                    frames, plan.residency.pool->layout.control_page_bytes,
                    control_arena_bytes)) ||
      !kernel::checked::mul(frames, output_page_bytes, output_arena_bytes) ||
      input_resource >= state.resources.size() ||
      output_resource >= state.resources.size() ||
      input_resource == output_resource ||
      (graph && (control_resource >= state.resources.size() ||
                 control_resource == input_resource ||
                 control_resource == output_resource)) ||
      !state.resources[input_resource].owned ||
      (graph && !state.resources[control_resource].owned) ||
      !state.resources[output_resource].owned ||
      state.resources[input_resource].bytes < input_arena_bytes ||
      (graph &&
       state.resources[control_resource].bytes != control_arena_bytes) ||
      state.resources[output_resource].bytes < output_arena_bytes ||
      state.resources[input_resource].buffer != expected_input ||
      (graph && state.resources[control_resource].buffer !=
                    plan.residency.pool->control[plan.residency.bank]) ||
      state.resources[output_resource].buffer != expected_output ||
      state.resources[output_resource].output == PipelineResource::no_output) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state.residency_input = input_resource;
  state.residency_control =
      graph ? control_resource : std::numeric_limits<std::uint32_t>::max();
  state.residency_output = output_resource;
  state.residency_input_page_bytes = input_page_bytes;
  state.residency_output_page_bytes = output_page_bytes;
  state.residency_graph_stage = plan.residency.graph_stage;
  state.residency_semantic = PipelineResidencySemantic::None;
  state.residency_semantic_ports = {};
  state.residency_semantic_port_count = 0u;
  state.residency_port_count = 0u;
  return Status::success();
}

} // namespace rund::compute::detail
