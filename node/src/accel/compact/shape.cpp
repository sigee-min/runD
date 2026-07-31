#include "shape.hpp"

namespace rund::node::accel::detail {

bool CompactShapeOk(const rund::kernel::CompactDesc &desc,
                    const rund::kernel::CompactPlan &plan,
                    const CompactBinds &bindings) noexcept {
  if (!rund::kernel::CompactPlanMatchesDesc(desc, plan) ||
      bindings.flags_handle == nullptr || bindings.output_handle == nullptr) {
    return false;
  }
  return PrimitiveResidentExactShapeOk(bindings.flags, plan.flag_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.output, plan.output_bytes,
                                       plan.output_capacity,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
