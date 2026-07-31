#include "shape.hpp"

namespace rund::node::accel::detail {

bool PartitionShapeOk(const rund::kernel::PartitionDesc &desc,
                      const rund::kernel::PartitionPlan &plan,
                      const PartitionBinds &bindings) noexcept {
  if (!rund::kernel::PartitionPlanMatchesDesc(desc, plan) ||
      bindings.flags_handle == nullptr || bindings.values_handle == nullptr ||
      bindings.output_handle == nullptr) {
    return false;
  }
  return PrimitiveResidentExactShapeOk(bindings.flags, plan.flag_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.values, plan.value_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.output, plan.value_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
