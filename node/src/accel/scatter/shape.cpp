#include "shape.hpp"

namespace rund::node::accel::detail {

bool ScatterShapeOk(const rund::kernel::ScatterDesc &desc,
                    const rund::kernel::ScatterPlan &plan,
                    const ScatterBinds &bindings) noexcept {
  if (!rund::kernel::ScatterPlanMatchesDesc(desc, plan) ||
      bindings.values_handle == nullptr || bindings.indices_handle == nullptr ||
      bindings.output_handle == nullptr) {
    return false;
  }
  return PrimitiveResidentExactShapeOk(bindings.values, plan.element_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.indices, plan.index_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.output, plan.element_bytes,
                                       plan.output_count,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
