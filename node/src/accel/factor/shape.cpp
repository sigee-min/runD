#include "shape.hpp"

namespace rund::node::accel::detail {

bool FactorShapeOk(const rund::kernel::FactorDesc &desc,
                   const rund::kernel::FactorPlan &plan,
                   const FactorBinds &bindings) noexcept {
  if (!rund::kernel::FactorPlanMatchesDesc(desc, plan) ||
      bindings.input_handle == nullptr || bindings.factor_handle == nullptr ||
      bindings.status_handle == nullptr) {
    return false;
  }
  if (!PrimitiveResidentExactShapeOk(bindings.input, plan.element_bytes,
                                     plan.input_count,
                                     rund::kernel::kResidentUsageRead) ||
      !PrimitiveResidentExactShapeOk(bindings.factor, plan.element_bytes,
                                     plan.factor_count,
                                     rund::kernel::kResidentUsageWrite) ||
      !PrimitiveResidentExactShapeOk(bindings.status, sizeof(rund::kernel::u32),
                                     plan.status_count,
                                     rund::kernel::kResidentUsageWrite)) {
    return false;
  }
  if (plan.op != rund::kernel::FactorOp::LU) {
    return true;
  }
  return bindings.aux_handle != nullptr &&
         PrimitiveResidentExactShapeOk(bindings.aux, sizeof(rund::kernel::u32),
                                       plan.aux_count,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
