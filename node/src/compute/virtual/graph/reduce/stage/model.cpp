#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

StageController::StageController(Stats &stats, residency::Pool &pool,
                                 Wavefront &wavefront,
                                 const std::uint64_t identity,
                                 const std::size_t terminal_stage) noexcept
    : stats_(stats), pool_(pool), wavefront_(wavefront), identity_(identity),
      terminal_stage_(terminal_stage) {}

} // namespace rund::compute::detail::graph_reduce
