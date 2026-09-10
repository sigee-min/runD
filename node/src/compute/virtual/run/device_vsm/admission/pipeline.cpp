#include "pipeline.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

std::shared_ptr<PipelineState>
primary_pipeline(VirtualPipelineState &state,
                 const VirtualRunProjection &run) noexcept {
  return run.graph_reduction && state.device_vsm_semantic_pipeline != nullptr
             ? state.device_vsm_semantic_pipeline
             : state.pipeline;
}

bool select_pipelines(
    VirtualPipelineState &state, const VirtualRunProjection &run,
    std::array<std::shared_ptr<PipelineState>, DeviceVsmPipelineCapacity>
        &pipelines,
    std::array<std::uint32_t, DeviceVsmPipelineCapacity> &stages,
    std::size_t &count) noexcept {
  pipelines = {};
  stages = {};
  count = 0u;
  const std::uint32_t terminal =
      run.graph_execution && state.pipeline != nullptr &&
              state.pipeline->residency != nullptr &&
              !state.pipeline->residency->tiled_graph().stages().empty()
          ? static_cast<std::uint32_t>(
                state.pipeline->residency->tiled_graph().stages().size() - 1u)
          : 0u;
  const bool graph_pointwise =
      run.graph_execution && !run.graph_reduction &&
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
  pipelines[0u] = primary_pipeline(state, run);
  pipelines[1u] = run.graph_execution ? graph_terminal_pipeline(state, 0u)
                                      : state.alternate_pipeline;
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
