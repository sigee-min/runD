#include "graph_persist_owner.hpp"

namespace rund::compute::detail::residency {

GraphPersistOwner::GraphPersistOwner(Authority &authority) noexcept
    : authority_(authority) {}

GraphPersistOwner Authority::graph_persists() noexcept {
  return GraphPersistOwner{*this};
}

} // namespace rund::compute::detail::residency
