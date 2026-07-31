#include "shape.hpp"

#include "../../primitive/shape.hpp"
#include <kernel/program/compute/segmented/reduce/plan.hpp>

namespace rund::node::accel::detail {

bool SegmentedReduceShapeOk(const rund::kernel::SegmentedReduceDesc &desc,
                            const rund::kernel::SegmentedReducePlan &plan,
                            const SegmentedReduceBinds &bindings) noexcept {
  if (!rund::kernel::SegmentedReducePlanMatchesDesc(desc, plan) ||
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
