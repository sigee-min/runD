#include "shape.hpp"

namespace rund::node::accel::detail {

bool SortShapeOk(const rund::kernel::SortDesc &desc,
                 const rund::kernel::SortPlan &plan,
                 const SortBinds &bindings) noexcept {
  const bool identity_values =
      plan.value == rund::kernel::SortValue::IdentityU32;
  const bool bounded =
      plan.count_source != rund::kernel::ComputeCountSource::Descriptor;
  if (!rund::kernel::SortPlanMatchesDesc(desc, plan) ||
      bindings.read_keys_handle == nullptr ||
      (!identity_values && bindings.read_values_handle == nullptr) ||
      (bounded && bindings.logical_count_handle == nullptr) ||
      bindings.write_keys_handle == nullptr ||
      bindings.write_values_handle == nullptr) {
    return false;
  }
  if (bounded && !PrimitiveResidentExactShapeOk(
                     bindings.logical_count,
                     rund::kernel::ComputeCountBytes(plan.count_source), 1u,
                     rund::kernel::kResidentUsageRead)) {
    return false;
  }
  if (!PrimitiveResidentExactShapeOk(bindings.read_keys, plan.key_bytes,
                                     plan.element_count,
                                     rund::kernel::kResidentUsageRead) ||
      !PrimitiveResidentExactShapeOk(bindings.write_keys, plan.key_bytes,
                                     plan.element_count,
                                     rund::kernel::kResidentUsageWrite) ||
      !PrimitiveResidentExactShapeOk(bindings.write_values, plan.value_bytes,
                                     plan.element_count,
                                     rund::kernel::kResidentUsageWrite)) {
    return false;
  }
  return identity_values ||
         PrimitiveResidentExactShapeOk(bindings.read_values, plan.value_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead);
}

} // namespace rund::node::accel::detail
