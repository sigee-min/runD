#include "plan.hpp"
#include "plan/build.hpp"

namespace rund::node::accel::detail {

RangePlan PlanRange(const RangeShape &shape,
                    const RangeCaps &capabilities) noexcept {
  return BuildRangePlan(shape, capabilities);
}

} // namespace rund::node::accel::detail
