#include "shape.hpp"

#include "../primitive/shape.hpp"
#include <kernel/program/compute/segmented/scan/plan.hpp>

namespace rund::node::accel::detail {

bool SegmentedScanShapeOk(const rund::kernel::SegmentedScanDesc &desc,
                          const rund::kernel::SegmentedScanPlan &plan,
                          const SegmentedScanBinds &bindings) noexcept {
  if (!rund::kernel::SegmentedScanPlanMatchesDesc(desc, plan) ||
      bindings.input_handle == nullptr || bindings.heads_handle == nullptr ||
      bindings.output_handle == nullptr) {
    return false;
  }
  return PrimitiveResidentExactShapeOk(bindings.input, plan.element_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.heads, plan.head_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.output, plan.element_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
