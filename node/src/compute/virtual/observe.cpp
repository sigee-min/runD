#include "observe/local.hpp"

#include "../device/residency/pool.hpp"
#include "../pipeline/local.hpp"

namespace rund::compute::detail {

bool valid_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return false;
  }
  if (state->geometry.route == VirtualRoute::MultiPointwise ||
      (state->geometry.route == VirtualRoute::Scan &&
       state->input_count > 1u)) {
    return virtual_observe::valid_poolless_device_vsm(*state);
  }
  const bool graph = state->geometry.route == VirtualRoute::GraphReduction ||
                     state->geometry.route == VirtualRoute::GraphPointwise;
  if (!virtual_observe::valid_input_authority(
          *state, 1u,
          graph ? residency::TiledGraphResourceCapacity - 2u : 1u) ||
      state->output == nullptr || state->pipeline == nullptr ||
      state->alternate_pipeline == nullptr ||
      state->pipeline->residency == nullptr ||
      state->pipeline->residency_pool == nullptr ||
      state->alternate_pipeline->residency != state->pipeline->residency ||
      state->alternate_pipeline->residency_pool !=
          state->pipeline->residency_pool ||
      state->pipeline->device != state->alternate_pipeline->device ||
      state->pipeline->residency_bank != 0u ||
      state->alternate_pipeline->residency_bank != 1u ||
      !state->pipeline->residency->identity() ||
      !valid_pipeline(state->pipeline) ||
      !valid_pipeline(state->alternate_pipeline)) {
    return false;
  }
  if (!graph) {
    return state->graph_pipelines.empty() &&
           state->graph_input_resource_count == 0u &&
           state->device_vsm_semantic_pipeline == nullptr;
  }
  return virtual_observe::valid_graph(*state);
}

} // namespace rund::compute::detail
