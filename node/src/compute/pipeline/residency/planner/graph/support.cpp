#include "internal.hpp"

#include <algorithm>

namespace rund::compute::detail::residency {

const TiledGraphResource *
find_graph_resource(const GraphPlanningState &state,
                    const std::uint32_t resource) noexcept {
  const auto found = std::lower_bound(
      state.resources.begin(), state.resources.end(), resource,
      [](const TiledGraphResource &value, const std::uint32_t id) {
        return value.resource < id;
      });
  return found == state.resources.end() || found->resource != resource
             ? nullptr
             : &*found;
}

std::size_t index_graph_resource(const GraphPlanningState &state,
                                 const std::uint32_t resource) noexcept {
  const TiledGraphResource *const found = find_graph_resource(state, resource);
  return found == nullptr
             ? state.resources.size()
             : static_cast<std::size_t>(found - state.resources.data());
}

} // namespace rund::compute::detail::residency
