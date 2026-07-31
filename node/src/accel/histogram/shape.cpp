#include "shape.hpp"

namespace rund::node::accel::detail {

bool HistogramShapeOk(const rund::kernel::HistogramDesc &desc,
                      const rund::kernel::HistogramPlan &plan,
                      const HistogramBinds &bindings) noexcept {
  if (!rund::kernel::HistogramPlanMatchesDesc(desc, plan) ||
      bindings.bins_handle == nullptr || bindings.counts_handle == nullptr) {
    return false;
  }
  return PrimitiveResidentExactShapeOk(bindings.bins, plan.index_bytes,
                                       plan.element_count,
                                       rund::kernel::kResidentUsageRead) &&
         PrimitiveResidentExactShapeOk(bindings.counts, plan.count_bytes,
                                       plan.bin_count,
                                       rund::kernel::kResidentUsageWrite);
}

} // namespace rund::node::accel::detail
