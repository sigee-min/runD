#include "local.hpp"

namespace program_compute_contract {

int CompactShape() {
  const rund::kernel::CompactPlan plan =
      rund::kernel::PlanCompact(U32Compact());
  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "ok");
  TEST_ASSERT(plan.element_count == 8u);
  TEST_ASSERT(plan.output_capacity == 8u);
  TEST_ASSERT(plan.flag_bytes == 4u);
  TEST_ASSERT(plan.output_bytes == 4u);
  TEST_ASSERT(plan.scan_temp_bytes == 32u);
  TEST_ASSERT(plan.status_bytes == 0u);
  TEST_ASSERT(plan.temp_bytes == 32u);
  TEST_ASSERT(plan.pass_count == 2u);
  TEST_ASSERT(rund::kernel::CompactPlanMatchesDesc(U32Compact(), plan));
  rund::kernel::CompactPlan forged = plan;
  ++forged.temp_bytes;
  TEST_ASSERT(!rund::kernel::CompactPlanMatchesDesc(U32Compact(), forged));

  rund::kernel::CompactDesc under_capacity = U32Compact();
  under_capacity.output_capacity = under_capacity.element_count - 1u;
  const rund::kernel::CompactPlan under_capacity_plan =
      rund::kernel::PlanCompact(under_capacity);
  TEST_ASSERT(under_capacity_plan.ok);
  TEST_ASSERT(under_capacity_plan.scan_temp_bytes == 32u);
  TEST_ASSERT(under_capacity_plan.status_bytes == 4u);
  TEST_ASSERT(under_capacity_plan.temp_bytes == 36u);
  TEST_ASSERT(under_capacity_plan.pass_count == 2u);
  return 0;
}

} // namespace program_compute_contract
