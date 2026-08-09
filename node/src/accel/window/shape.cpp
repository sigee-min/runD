#include "shape.hpp"

namespace rund::node::accel::detail {

bool WindowShapeOk(const rund::kernel::WindowDesc &desc,
                   const rund::kernel::WindowPlan &plan,
                   const RangeBinds &bindings) noexcept {
  if (!rund::kernel::WindowPlanMatchesDesc(desc, plan) ||
      bindings.input_handle == nullptr || bindings.output_handle == nullptr) {
    return false;
  }
  return PrimitiveResidentExactShapeOk(bindings.input, plan.element_bytes,
                                       plan.input_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.output, plan.element_bytes,
                                       plan.output_count,
                                       rund::kernel::kResidentUsageWrite) &&
         !ResidentOverlap(*bindings.input, *bindings.output);
}

} // namespace rund::node::accel::detail
