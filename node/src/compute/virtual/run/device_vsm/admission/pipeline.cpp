#include "pipeline.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

bool select_pipelines(
    VirtualPipelineState &state, const VirtualRunTopology topology,
    std::array<std::shared_ptr<PipelineState>, DeviceVsmPipelineCapacity>
        &pipelines,
    std::array<std::uint32_t, DeviceVsmPipelineCapacity> &stages,
    std::size_t &count) noexcept {
  pipelines = {};
  stages = {};
  count = 0u;
  const bool graph = topology == VirtualRunTopology::GraphPointwise ||
                     topology == VirtualRunTopology::GraphReduction;
  const std::uint32_t terminal =
      graph && state.pipeline != nullptr &&
              state.pipeline->residency != nullptr &&
              !state.pipeline->residency->tiled_graph().stages().empty()
          ? static_cast<std::uint32_t>(
                state.pipeline->residency->tiled_graph().stages().size() - 1u)
          : 0u;
  const bool graph_pointwise =
      topology == VirtualRunTopology::GraphPointwise &&
      state.geometry.route == VirtualRoute::GraphPointwise;
  if (graph_pointwise) {
    const std::size_t stage_count = static_cast<std::size_t>(terminal) + 1u;
    if (stage_count < 2u || stage_count > DeviceVsmPipelineCapacity ||
        state.graph_pipelines.size() !=
            stage_count * residency::Pool::BankCount) {
      return false;
    }
    for (std::size_t stage = 0u; stage < stage_count; ++stage) {
      pipelines[stage] = graph_stage_pipeline(state, stage, 0u);
      stages[stage] = static_cast<std::uint32_t>(stage);
    }
    count = stage_count;
    return true;
  }
  pipelines[0u] = topology == VirtualRunTopology::GraphReduction &&
                          state.device_vsm_semantic_pipeline != nullptr
                      ? state.device_vsm_semantic_pipeline
                      : state.pipeline;
  pipelines[1u] =
      graph ? graph_terminal_pipeline(state, 0u) : state.alternate_pipeline;
  stages[0u] = 0u;
  stages[1u] = terminal;
  count = 2u;
  return pipelines[0u] != nullptr && pipelines[1u] != nullptr;
}

namespace admission_detail {

bool pipeline_authority(const std::shared_ptr<PipelineState> &selected_primary,
                        const std::shared_ptr<PipelineState> &second,
                        const VirtualRunProjection &run,
                        const bool multi_pointwise,
                        const bool multi_scan) noexcept {
  return selected_primary != nullptr && second != nullptr &&
         ((multi_pointwise || multi_scan)
              ? selected_primary->residency == nullptr &&
                    second->residency == nullptr &&
                    selected_primary->residency_pool == nullptr &&
                    second->residency_pool == nullptr &&
                    run.frame_capacity == (multi_scan ? 2u : 1u)
              : selected_primary->residency_pool != nullptr &&
                    selected_primary->residency_pool == second->residency_pool);
}

} // namespace admission_detail

} // namespace rund::compute::detail::device_vsm_product_detail
