#pragma once

#include "model/plan.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] RangePlan PlanRange(const RangeShape &shape,
                                  const RangeCaps &capabilities) noexcept;

} // namespace rund::node::accel::detail
