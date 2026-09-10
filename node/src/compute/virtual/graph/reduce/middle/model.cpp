#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

MiddleController::MiddleController(
    VirtualPipelineState &state, const VirtualRunProjection &run, Stats &stats,
    residency::Pool &pool, const residency::TiledGraphPlan &graph,
    Wavefront &wavefront, PrefetchController &prefetch, StageController &stages,
    const std::uint64_t identity, const std::uint32_t capacity,
    const std::size_t terminal_stage) noexcept
    : state_(state), run_(run), stats_(stats), pool_(pool),
      authority_(pool.authority()), graph_(graph), wavefront_(wavefront),
      prefetch_(prefetch), stages_(stages), identity_(identity),
      capacity_(capacity), terminal_stage_(terminal_stage) {}

} // namespace rund::compute::detail::graph_reduce
