#pragma once

#include "stage.hpp"

namespace rund::compute::detail::graph_reduce {

class FailLog;

[[nodiscard]] Status finish_prefix(residency::Authority &,
                                   const residency::TiledGraphPlan &,
                                   Wavefront &, StageController &, Ticket &,
                                   FailLog &, bool &child_poison) noexcept;

} // namespace rund::compute::detail::graph_reduce
