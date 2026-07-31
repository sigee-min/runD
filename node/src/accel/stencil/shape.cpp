#include "shape.hpp"

namespace rund::node::accel::detail {

bool StencilShapeOk(const rund::kernel::StencilDesc &desc,
                    const rund::kernel::StencilPlan &plan,
                    const StencilBinds &bindings) noexcept {
  if (!rund::kernel::StencilPlanMatchesDesc(desc, plan) ||
      bindings.input_handle == nullptr || bindings.output_handle == nullptr) {
    return false;
  }
  return PrimitiveResidentExactShapeOk(bindings.input, plan.element_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.output, plan.element_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
