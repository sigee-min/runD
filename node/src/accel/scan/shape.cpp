#include "shape.hpp"

#include "../primitive/shape.hpp"

#include <kernel/program/compute/scan/plan.hpp>

namespace rund::node::accel::detail {

bool ScanShapeOk(const rund::kernel::ScanDesc &desc,
                 const rund::kernel::ScanPlan &plan) noexcept {
  return rund::kernel::ScanPlanMatchesDesc(desc, plan);
}

bool ScanResidentShapeOk(const rund::kernel::ScanPlan &plan,
                         const ScanBinds &bindings) noexcept {
  if (bindings.input_handle == nullptr || bindings.output_handle == nullptr ||
      !PrimitiveResidentAtLeastShapeOk(bindings.input, plan.element_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) ||
      !PrimitiveResidentAtLeastShapeOk(bindings.output, plan.element_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageWrite)) {
    return false;
  }
  return plan.count_source == rund::kernel::ComputeCountSource::Descriptor ||
         (bindings.logical_count_handle != nullptr &&
          PrimitiveResidentExactShapeOk(
              bindings.logical_count,
              rund::kernel::ComputeCountBytes(plan.count_source), 1u,
              rund::kernel::kResidentUsageRead));
}

} // namespace rund::node::accel::detail
