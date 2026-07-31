#include "shape.hpp"

namespace rund::node::accel::detail {

bool GatherShapeOk(const rund::kernel::GatherDesc &desc,
                   const rund::kernel::GatherPlan &plan,
                   const GatherBinds &bindings) noexcept {
  if (!rund::kernel::GatherPlanMatchesDesc(desc, plan) ||
      bindings.values_handle == nullptr || bindings.indices_handle == nullptr ||
      bindings.output_handle == nullptr ||
      (plan.count_source != rund::kernel::ComputeCountSource::Descriptor &&
       bindings.logical_count_handle == nullptr)) {
    return false;
  }
  return PrimitiveResidentAtLeastShapeOk(bindings.values, plan.element_bytes,
                                         plan.source_count,
                                         rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.indices, plan.index_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         (plan.count_source == rund::kernel::ComputeCountSource::Descriptor ||
          PrimitiveResidentExactShapeOk(
              bindings.logical_count,
              rund::kernel::ComputeCountBytes(plan.count_source), 1u,
              rund::kernel::kResidentUsageRead)) &&
         PrimitiveResidentExactShapeOk(bindings.output, plan.element_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
