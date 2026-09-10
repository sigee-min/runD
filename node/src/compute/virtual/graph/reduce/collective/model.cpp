#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

CollectiveController::CollectiveController(
    const VirtualRunProjection &run, Stats &stats, residency::Pool &pool,
    const residency::TiledGraphPlan &graph, Wavefront &wavefront,
    StageController &stages, const std::size_t terminal_stage,
    FailLog &failure) noexcept
    : run_(run), stats_(stats), pool_(pool), authority_(pool.authority()),
      graph_(graph), wavefront_(wavefront), stages_(stages),
      terminal_stage_(terminal_stage), failure_(failure) {}

} // namespace rund::compute::detail::graph_reduce
