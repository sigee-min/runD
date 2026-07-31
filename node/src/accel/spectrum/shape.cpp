#include "shape.hpp"

namespace rund::node::accel::detail {

bool SpectrumShapeOk(const rund::kernel::SpectrumDesc &desc,
                     const rund::kernel::SpectrumPlan &plan,
                     const SpectrumBinds &bindings) noexcept {
  if (!rund::kernel::SpectrumPlanMatchesDesc(desc, plan) ||
      bindings.input_handle == nullptr || bindings.values_handle == nullptr ||
      bindings.status_handle == nullptr) {
    return false;
  }
  if (!PrimitiveResidentExactShapeOk(bindings.input, plan.element_bytes,
                                     plan.input_count,
                                     rund::kernel::kResidentUsageRead) ||
      !PrimitiveResidentExactShapeOk(bindings.values, plan.element_bytes,
                                     plan.value_count,
                                     rund::kernel::kResidentUsageWrite) ||
      !PrimitiveResidentExactShapeOk(bindings.status, sizeof(rund::kernel::u32),
                                     plan.status_count,
                                     rund::kernel::kResidentUsageWrite)) {
    return false;
  }
  if (plan.vector_count == 0u) {
    return true;
  }
  return bindings.vectors_handle != nullptr &&
         PrimitiveResidentExactShapeOk(bindings.vectors, plan.element_bytes,
                                       plan.vector_count,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
