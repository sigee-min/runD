#include "local.hpp"

#include <kernel/program/compute/plan.hpp>

namespace rund::node::accel::detail {

bool ComputePlanHeaderValid(
    const rund::kernel::ComputePlan& plan,
    const rund::kernel::ComputeApi expected_api) noexcept {
  return rund::kernel::ComputePlanShapeValid(plan) &&
         plan.api == expected_api &&
         rund::kernel::ComputePlanScalarValid(plan);
}

}  // namespace rund::node::accel::detail
