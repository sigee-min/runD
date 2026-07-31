#include "local.hpp"

namespace rund::node::accel::detail {

bool FrozenCapsAdmitPlan(const rund::kernel::ComputeCaps& caps,
                            const rund::kernel::ComputePlan& plan) noexcept {
  return caps.ok &&
         ReasonIsOk(caps.reason) &&
         plan.api == caps.api &&
         plan.staging_bytes <= caps.staging_bytes &&
         plan.dispatch_window_tiles <= caps.max_window_tiles;
}

}  // namespace rund::node::accel::detail
