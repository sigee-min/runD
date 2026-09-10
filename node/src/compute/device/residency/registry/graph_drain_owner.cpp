#include "graph_drain_owner.hpp"

namespace rund::compute::detail::residency {

GraphDrainOwner::GraphDrainOwner(Authority &authority) noexcept
    : authority_(authority) {}

GraphDrainOwner Authority::graph_drains() noexcept {
  return GraphDrainOwner{*this};
}

} // namespace rund::compute::detail::residency
