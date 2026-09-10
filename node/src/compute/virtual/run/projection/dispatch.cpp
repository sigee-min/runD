#include "internal.hpp"

namespace rund::compute::detail {

bool project_virtual_run(VirtualPipelineState &state,
                         const std::uint64_t active_count,
                         VirtualRunProjection &projection) noexcept {
  projection = {};
  if (state.geometry.route == VirtualRoute::GraphReduction ||
      state.geometry.route == VirtualRoute::GraphPointwise) {
    return project_virtual_graph_run(state, active_count, projection);
  }
  if (state.geometry.route == VirtualRoute::MultiPointwise ||
      (state.geometry.route == VirtualRoute::Scan && state.input_count > 1u)) {
    return project_virtual_multi_run(state, active_count, projection);
  }
  return project_virtual_ordinary_run(state, active_count, projection);
}

} // namespace rund::compute::detail
