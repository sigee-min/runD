#include "local.hpp"

#include "../../device/residency/pool.hpp"
#include "../../pipeline/local.hpp"

#include <algorithm>
#include <limits>

namespace rund::compute::detail::virtual_observe {
namespace {

[[nodiscard]] bool
valid_semantic_pipeline(const VirtualPipelineState &state,
                        const PipelineState &pipeline) noexcept {
  if (pipeline.residency == nullptr || pipeline.residency_pool == nullptr ||
      pipeline.residency_stage != PipelineResidencyStage::Graph ||
      pipeline.residency_graph_stage != 0u ||
      pipeline.residency_bank >= residency::Pool::BankCount ||
      pipeline.residency_semantic_port_count != state.input_count + 1u ||
      pipeline.residency_semantic_port_count >
          pipeline.residency_semantic_ports.size() ||
      pipeline.residency_port_count != pipeline.residency_semantic_port_count ||
      pipeline.residency_control >= pipeline.resources.size()) {
    return false;
  }
  const PipelineResidencySemantic expected_semantic =
      state.geometry.route == VirtualRoute::GraphReduction
          ? PipelineResidencySemantic::GraphReduction
          : PipelineResidencySemantic::GraphPointwise;
  if (pipeline.residency_semantic != expected_semantic ||
      pipeline.residency != state.pipeline->residency ||
      pipeline.residency_pool != state.pipeline->residency_pool) {
    return false;
  }
  const residency::TiledGraphPlan &graph = pipeline.residency->tiled_graph();
  if (graph.stages().empty() ||
      pipeline.residency_semantic_port_count != state.input_count + 1u) {
    return false;
  }
  std::uint32_t expected_output = std::numeric_limits<std::uint32_t>::max();
  const residency::TiledGraphStage &terminal = graph.stages().back();
  for (const residency::TiledGraphPort &port : terminal.ports) {
    if ((expected_semantic == PipelineResidencySemantic::GraphReduction &&
         port.access == residency::Access::Read) ||
        (expected_semantic == PipelineResidencySemantic::GraphPointwise &&
         port.access == residency::Access::Write)) {
      expected_output = port.resource;
      break;
    }
  }
  if (expected_output == std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  for (std::size_t index = 0u; index < pipeline.residency_semantic_port_count;
       ++index) {
    const PipelineResidencySemanticPort &semantic =
        pipeline.residency_semantic_ports[index];
    const bool read = index + 1u < pipeline.residency_semantic_port_count;
    const std::uint32_t expected_resource =
        read ? state.graph_input_resources[index] : expected_output;
    if (semantic.resource != expected_resource ||
        semantic.program_port != (read ? index : 0u) ||
        semantic.access !=
            (read ? residency::Access::Read : residency::Access::Write) ||
        pipeline.residency_ports[index].resource != semantic.resource ||
        pipeline.residency_ports[index].program_port != semantic.program_port ||
        pipeline.residency_ports[index].access != semantic.access ||
        pipeline.residency_ports[index].pipeline_resource >=
            pipeline.resources.size()) {
      return false;
    }
    const residency::TiledGraphResource *const resource =
        graph.resource(semantic.resource);
    const residency::PoolPhysicalOwner *const owner =
        resource == nullptr
            ? nullptr
            : pipeline.residency_pool->graph_owner(resource->physical_id);
    if (owner == nullptr ||
        owner->buffers[pipeline.residency_bank] == nullptr ||
        pipeline.resources[pipeline.residency_ports[index].pipeline_resource]
                .buffer != owner->buffers[pipeline.residency_bank] ||
        pipeline.resources[pipeline.residency_ports[index].pipeline_resource]
                .type != resource->type ||
        pipeline.resources[pipeline.residency_ports[index].pipeline_resource]
                .format != resource->format ||
        pipeline.residency_ports[index].page_bytes != resource->page_bytes) {
      return false;
    }
  }
  const PipelineResource &control =
      pipeline.resources[pipeline.residency_control];
  return control.buffer ==
             pipeline.residency_pool->control[pipeline.residency_bank] &&
         control.output == PipelineResource::no_output;
}

} // namespace

bool valid_graph(const VirtualPipelineState &state) noexcept {
  const residency::TiledGraphPlan &plan =
      state.pipeline->residency->tiled_graph();
  if (plan.stages().size() < 2u ||
      state.graph_input_resource_count != state.input_count ||
      plan.stages().size() > std::numeric_limits<std::size_t>::max() /
                                 residency::Pool::BankCount ||
      state.graph_pipelines.size() !=
          plan.stages().size() * residency::Pool::BankCount) {
    return false;
  }
  for (std::size_t index = 0u; index < state.graph_input_resource_count;
       ++index) {
    const std::uint32_t resource = state.graph_input_resources[index];
    const residency::TiledGraphResource *const row = plan.resource(resource);
    if (resource == 0u || row == nullptr ||
        row->kind != residency::GraphResourceKind::ExternalInput ||
        row->persistence != residency::ResourcePersistence::Backing ||
        std::find(state.graph_input_resources.begin(),
                  state.graph_input_resources.begin() + index,
                  resource) != state.graph_input_resources.begin() + index) {
      return false;
    }
  }
  for (std::size_t stage = 0u; stage < plan.stages().size(); ++stage) {
    for (std::size_t bank = 0u; bank < residency::Pool::BankCount; ++bank) {
      const std::shared_ptr<PipelineState> pipeline =
          graph_stage_pipeline(state, stage, bank);
      if (pipeline == nullptr || !valid_pipeline(pipeline) ||
          pipeline->residency != state.pipeline->residency ||
          pipeline->residency_pool != state.pipeline->residency_pool ||
          pipeline->device != state.pipeline->device ||
          pipeline->residency_stage != PipelineResidencyStage::Graph ||
          pipeline->residency_graph_stage != stage ||
          pipeline->residency_bank != bank) {
        return false;
      }
    }
  }
  const std::shared_ptr<PipelineState> first_pipeline =
      graph_stage_pipeline(state, 0u, 0u);
  const std::shared_ptr<PipelineState> alternate_first_pipeline =
      graph_stage_pipeline(state, 0u, 1u);
  const std::shared_ptr<PipelineState> terminal_pipeline =
      graph_terminal_pipeline(state, 0u);
  const std::shared_ptr<PipelineState> alternate_terminal_pipeline =
      graph_terminal_pipeline(state, 1u);
  const std::size_t terminal_stage = plan.stages().size() - 1u;
  const std::shared_ptr<PipelineState> &semantic =
      state.device_vsm_semantic_pipeline;
  const bool graph_reduction =
      state.geometry.route == VirtualRoute::GraphReduction;
  const bool semantic_valid =
      (!graph_reduction && semantic == nullptr) ||
      (graph_reduction &&
       (semantic == nullptr ||
        (plan.stages().size() > 2u &&
         state.pipeline->device->backend != Backend::Cpu &&
         semantic != state.pipeline && valid_pipeline(semantic) &&
         semantic->device == state.pipeline->device &&
         semantic->residency == state.pipeline->residency &&
         semantic->residency_pool == state.pipeline->residency_pool &&
         semantic->residency_stage == PipelineResidencyStage::Graph &&
         semantic->residency_graph_stage == 0u &&
         semantic->residency_bank == 0u)));
  const bool semantic_owner_valid =
      semantic == nullptr ||
      (graph_reduction && valid_semantic_pipeline(state, *semantic));
  return semantic_valid && semantic_owner_valid &&
         state.pipeline == first_pipeline &&
         state.alternate_pipeline == alternate_first_pipeline &&
         state.pipeline->residency_stage == PipelineResidencyStage::Graph &&
         state.pipeline->residency_graph_stage == 0u &&
         state.alternate_pipeline->residency_stage ==
             PipelineResidencyStage::Graph &&
         state.alternate_pipeline->residency_graph_stage == 0u &&
         terminal_pipeline != nullptr &&
         alternate_terminal_pipeline != nullptr &&
         terminal_pipeline->residency == state.pipeline->residency &&
         alternate_terminal_pipeline->residency == state.pipeline->residency &&
         terminal_pipeline->residency_pool == state.pipeline->residency_pool &&
         alternate_terminal_pipeline->residency_pool ==
             state.pipeline->residency_pool &&
         terminal_pipeline->device == state.pipeline->device &&
         alternate_terminal_pipeline->device == state.pipeline->device &&
         terminal_pipeline->residency_bank == 0u &&
         alternate_terminal_pipeline->residency_bank == 1u &&
         terminal_pipeline->residency_stage == PipelineResidencyStage::Graph &&
         terminal_pipeline->residency_graph_stage == terminal_stage &&
         alternate_terminal_pipeline->residency_stage ==
             PipelineResidencyStage::Graph &&
         alternate_terminal_pipeline->residency_graph_stage == terminal_stage;
}

} // namespace rund::compute::detail::virtual_observe
