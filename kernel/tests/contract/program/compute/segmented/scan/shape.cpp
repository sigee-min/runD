#include "local.hpp"

namespace program_compute_contract {

int SegmentedScanShape() {
  const rund::kernel::SegmentedScanPlan u32_plan =
      rund::kernel::PlanSegmentedScan(U32SegmentedScan());
  TEST_ASSERT(u32_plan.ok);
  TEST_ASSERT(std::string_view{u32_plan.reason} == "ok");
  TEST_ASSERT(u32_plan.op == rund::kernel::SegmentedScanOp::ExclusiveSum);
  TEST_ASSERT(u32_plan.element == rund::kernel::SegmentedScanElement::U32);
  TEST_ASSERT(u32_plan.element_count == 8u);
  TEST_ASSERT(u32_plan.element_bytes == 4u);
  TEST_ASSERT(u32_plan.head_bytes == 4u);
  TEST_ASSERT(u32_plan.block_size == 4u);
  TEST_ASSERT(u32_plan.block_count == 2u);
  TEST_ASSERT(u32_plan.pass_count == 2u);
  TEST_ASSERT(u32_plan.temp_value_bytes == 32u);
  TEST_ASSERT(u32_plan.temp_head_bytes == 32u);
  TEST_ASSERT(u32_plan.temp_bytes == 64u);
  TEST_ASSERT(
      rund::kernel::SegmentedScanPlanMatchesDesc(U32SegmentedScan(), u32_plan));
  rund::kernel::SegmentedScanPlan forged = u32_plan;
  ++forged.temp_head_bytes;
  TEST_ASSERT(
      !rund::kernel::SegmentedScanPlanMatchesDesc(U32SegmentedScan(), forged));

  rund::kernel::SegmentedScanDesc u64_desc = U32SegmentedScan();
  u64_desc.element = rund::kernel::SegmentedScanElement::U64;
  u64_desc.element_count = 9u;
  const rund::kernel::SegmentedScanPlan u64_plan =
      rund::kernel::PlanSegmentedScan(u64_desc);
  TEST_ASSERT(u64_plan.ok);
  TEST_ASSERT(u64_plan.element_bytes == 8u);
  TEST_ASSERT(u64_plan.temp_value_bytes == 72u);
  TEST_ASSERT(u64_plan.temp_head_bytes == 36u);
  TEST_ASSERT(u64_plan.temp_bytes == 108u);
  return 0;
}

} // namespace program_compute_contract
