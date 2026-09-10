#include "graph_promote_owner.hpp"

namespace rund::compute::detail::residency {

GraphPromoteOwner::GraphPromoteOwner(Authority &authority) noexcept
    : authority_(authority) {}

GraphPromoteOwner Authority::graph_promotes() noexcept {
  return GraphPromoteOwner{*this};
}

} // namespace rund::compute::detail::residency
