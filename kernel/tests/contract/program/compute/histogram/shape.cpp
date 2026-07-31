#include "local.hpp"

namespace program_compute_contract {

int HistogramShape() {
  const rund::kernel::HistogramPlan plan =
      rund::kernel::PlanHistogram(U32Histogram());
  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "ok");
  TEST_ASSERT(plan.index == rund::kernel::HistogramIndex::U32);
  TEST_ASSERT(plan.count == rund::kernel::HistogramCount::U32);
  TEST_ASSERT(plan.element_count == 8u);
  TEST_ASSERT(plan.bin_count == 4u);
  TEST_ASSERT(plan.index_bytes == 4u);
  TEST_ASSERT(plan.count_bytes == 4u);
  TEST_ASSERT(plan.input_bytes == 32u);
  TEST_ASSERT(plan.output_bytes == 16u);
  TEST_ASSERT(plan.status_bytes == 4u);
  TEST_ASSERT(plan.temp_bytes == 4u);
  TEST_ASSERT(plan.pass_count == 2u);
  TEST_ASSERT(rund::kernel::HistogramPlanMatchesDesc(U32Histogram(), plan));
  rund::kernel::HistogramPlan forged = plan;
  ++forged.input_bytes;
  TEST_ASSERT(!rund::kernel::HistogramPlanMatchesDesc(U32Histogram(), forged));
  return 0;
}

} // namespace program_compute_contract
