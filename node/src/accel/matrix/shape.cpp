#include "shape.hpp"

namespace rund::node::accel::detail {

bool MatrixShapeOk(const rund::kernel::MatrixDesc &desc,
                   const rund::kernel::MatrixPlan &plan,
                   const MatrixBinds &bindings) noexcept {
  if (!rund::kernel::MatrixPlanMatchesDesc(desc, plan) ||
      bindings.left_handle == nullptr || bindings.output_handle == nullptr) {
    return false;
  }
  if (!PrimitiveResidentExactShapeOk(bindings.left, plan.element_bytes,
                                     plan.left_count,
                                     rund::kernel::kResidentUsageRead) ||
      !PrimitiveResidentExactShapeOk(bindings.output, plan.element_bytes,
                                     plan.output_count,
                                     rund::kernel::kResidentUsageWrite)) {
    return false;
  }
  if (plan.op == rund::kernel::MatrixOp::Transpose) {
    return true;
  }
  return bindings.right_handle != nullptr &&
         PrimitiveResidentExactShapeOk(bindings.right, plan.element_bytes,
                                       plan.right_count,
                                       rund::kernel::kResidentUsageRead);
}

} // namespace rund::node::accel::detail
