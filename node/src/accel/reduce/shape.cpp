#include "shape.hpp"

namespace rund::node::accel::detail {

bool ReduceShapeOk(const rund::kernel::ReduceDesc &desc,
                   const rund::kernel::ReducePlan &plan,
                   const ReduceBinds &bindings) noexcept {
  const bool bounded =
      plan.count_source != rund::kernel::ComputeCountSource::Descriptor;
  if (!rund::kernel::ReducePlanMatchesDesc(desc, plan) ||
      bindings.input_handle == nullptr || bindings.output_handle == nullptr ||
      (bounded && bindings.logical_count_handle == nullptr) ||
      plan.block_size > 1024u) {
    return false;
  }
  const bool count_ok =
      !bounded || PrimitiveResidentExactShapeOk(
                      bindings.logical_count,
                      rund::kernel::ComputeCountBytes(plan.count_source), 1u,
                      rund::kernel::kResidentUsageRead);
  return count_ok &&
         PrimitiveResidentExactShapeOk(bindings.input, plan.element_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.output, plan.element_bytes, 1u,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
