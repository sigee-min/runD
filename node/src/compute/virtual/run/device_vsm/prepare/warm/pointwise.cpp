#include "local.hpp"

#include "../graph_wavefront.hpp"

namespace rund::compute::detail::device_vsm_product_detail::warm_detail {

bool validate_pointwise(const VirtualPipelineState &state,
                        const VirtualRunProjection &run,
                        const DeviceVsmProductOwner &owner) noexcept {
  if (owner.pipeline_count < 2u || state.pipeline == nullptr ||
      state.pipeline->residency == nullptr) {
    return false;
  }
  node::accel::detail::DeviceVsmGraphWavefrontProof fresh_wavefront{};
  node::accel::detail::DeviceVsmPageMap fresh_map{};
  if (!project_graph_wavefront(state.pipeline->residency->tiled_graph(),
                               page_count(run), owner.pipeline_stages[0u],
                               owner.pipeline_stages[owner.pipeline_count - 1u],
                               fresh_wavefront) ||
      !project_graph_page_map(state.pipeline->residency->tiled_graph(),
                              {state.graph_input_resources.data(),
                               state.graph_input_resource_count},
                              fresh_map)) {
    return false;
  }
  return same_wavefront(fresh_wavefront,
                        owner.proof->graph_pointwise.wavefront) &&
         node::accel::detail::device_vsm_page_map_equal(
             fresh_map, owner.proof->graph_pointwise.page_map);
}

} // namespace rund::compute::detail::device_vsm_product_detail::warm_detail
