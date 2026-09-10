#include "../internal.hpp"

namespace rund::compute::detail::residency::execution {

bool SlidingInvocation::predecessors(
    const SlidingProjection &projection,
    const std::span<SlidingCoordinate, SlidingPredecessorCapacity> storage,
    std::size_t &count) const noexcept {
  count = 0u;
  if (!*this || projection.coordinate.ordinal >= this->count()) {
    return false;
  }
  if (topology_ == SlidingTopology::Direct) {
    if (projection.coordinate.ordinal >= BankCapacity) {
      const std::uint64_t prior = projection.coordinate.ordinal - BankCapacity;
      storage[count++] =
          SlidingCoordinate{.ordinal = prior, .batch = prior, .stage = 0u};
    }
    return true;
  }
  std::array<TiledGraphDependency, TiledGraphDependencyCapacity> sealed{};
  if (!graph_.predecessors(projection.coordinate.batch,
                           projection.coordinate.stage, sealed, count)) {
    count = 0u;
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    storage[index] = SlidingCoordinate{.ordinal = sealed[index].ordinal,
                                       .batch = sealed[index].batch,
                                       .stage = sealed[index].stage};
  }
  return true;
}

} // namespace rund::compute::detail::residency::execution
