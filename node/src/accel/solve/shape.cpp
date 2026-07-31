#include "shape.hpp"

namespace rund::node::accel::detail {

bool SolveShapeOk(const rund::kernel::SolveDesc &desc,
                  const rund::kernel::SolvePlan &plan,
                  const SolveBinds &bindings) noexcept {
  if (!rund::kernel::SolvePlanMatchesDesc(desc, plan) ||
      bindings.primary_handle == nullptr || bindings.rhs_handle == nullptr ||
      bindings.output_handle == nullptr || bindings.status_handle == nullptr) {
    return false;
  }
  const std::uint64_t primary_count =
      plan.input == rund::kernel::SolveInput::Factor ? plan.factor_count
                                                     : plan.matrix_count;
  if (!PrimitiveResidentExactShapeOk(bindings.primary, plan.element_bytes,
                                     primary_count,
                                     rund::kernel::kResidentUsageRead) ||
      !PrimitiveResidentExactShapeOk(bindings.rhs, plan.element_bytes,
                                     plan.rhs_count,
                                     rund::kernel::kResidentUsageRead) ||
      !PrimitiveResidentExactShapeOk(bindings.output, plan.element_bytes,
                                     plan.output_count,
                                     rund::kernel::kResidentUsageWrite) ||
      !PrimitiveResidentExactShapeOk(bindings.status, sizeof(rund::kernel::u32),
                                     plan.status_count,
                                     rund::kernel::kResidentUsageWrite)) {
    return false;
  }
  if (plan.input != rund::kernel::SolveInput::Factor ||
      plan.factor != rund::kernel::FactorOp::LU) {
    return true;
  }
  return bindings.aux_handle != nullptr &&
         PrimitiveResidentExactShapeOk(bindings.aux, sizeof(rund::kernel::u32),
                                       plan.aux_count,
                                       rund::kernel::kResidentUsageRead);
}

} // namespace rund::node::accel::detail
