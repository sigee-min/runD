#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

SupplyController::SupplyController(const VirtualRunProjection &run,
                                   Stats &stats, residency::Pool &pool,
                                   const residency::TiledGraphPlan &graph,
                                   Wavefront &wavefront,
                                   PrefetchController &prefetch,
                                   FailLog &failure) noexcept
    : run_(run), stats_(stats), pool_(pool), authority_(pool.authority()),
      graph_(graph), wavefront_(wavefront), prefetch_(prefetch),
      failure_(failure) {}

} // namespace rund::compute::detail::graph_reduce
