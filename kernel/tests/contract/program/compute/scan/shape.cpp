#include "local.hpp"

namespace program_compute_contract {

int ScanShape() {
  const rund::kernel::ScanPlan u32_plan = rund::kernel::PlanScan(U32Scan());
  TEST_ASSERT(u32_plan.ok);
  TEST_ASSERT(std::string_view{u32_plan.reason} == "ok");
  TEST_ASSERT(u32_plan.op == rund::kernel::ScanOp::ExclusiveSum);
  TEST_ASSERT(u32_plan.element == rund::kernel::ScanElement::U32);
  TEST_ASSERT(u32_plan.element_count == 8u);
  TEST_ASSERT(u32_plan.element_bytes == 4u);
  TEST_ASSERT(u32_plan.block_size == 4u);
  TEST_ASSERT(u32_plan.block_count == 2u);
  TEST_ASSERT(u32_plan.pass_count == 2u);
  TEST_ASSERT(u32_plan.temp_bytes == 32u);
  TEST_ASSERT(u32_plan.count_source ==
              rund::kernel::ComputeCountSource::Descriptor);
  TEST_ASSERT(rund::kernel::ScanPlanMatchesDesc(U32Scan(), u32_plan));
  rund::kernel::ScanPlan forged = u32_plan;
  ++forged.temp_bytes;
  TEST_ASSERT(!rund::kernel::ScanPlanMatchesDesc(U32Scan(), forged));

  rund::kernel::ScanDesc bounded_desc = U32Scan();
  bounded_desc.count_source = rund::kernel::ComputeCountSource::BufferU64;
  const rund::kernel::ScanPlan bounded = rund::kernel::PlanScan(bounded_desc);
  TEST_ASSERT(bounded.ok);
  TEST_ASSERT(bounded.element_count == u32_plan.element_count);
  TEST_ASSERT(bounded.temp_bytes == u32_plan.temp_bytes);
  TEST_ASSERT(bounded.count_source ==
              rund::kernel::ComputeCountSource::BufferU64);

  rund::kernel::ScanDesc u64_desc = U32Scan();
  u64_desc.element = rund::kernel::ScanElement::U64;
  u64_desc.element_count = 9u;
  u64_desc.block_size = 4u;
  const rund::kernel::ScanPlan u64_plan = rund::kernel::PlanScan(u64_desc);
  TEST_ASSERT(u64_plan.ok);
  TEST_ASSERT(u64_plan.element == rund::kernel::ScanElement::U64);
  TEST_ASSERT(u64_plan.element_bytes == 8u);
  TEST_ASSERT(u64_plan.block_count == 3u);
  TEST_ASSERT(u64_plan.pass_count == 2u);
  TEST_ASSERT(u64_plan.temp_bytes == 72u);

  rund::kernel::ScanDesc inclusive_desc = U32Scan();
  inclusive_desc.op = rund::kernel::ScanOp::InclusiveSum;
  const rund::kernel::ScanPlan inclusive_plan =
      rund::kernel::PlanScan(inclusive_desc);
  TEST_ASSERT(inclusive_plan.ok);
  TEST_ASSERT(inclusive_plan.op == rund::kernel::ScanOp::InclusiveSum);
  TEST_ASSERT(inclusive_plan.element == rund::kernel::ScanElement::U32);
  TEST_ASSERT(inclusive_plan.element_count == 8u);
  TEST_ASSERT(inclusive_plan.block_count == 2u);
  TEST_ASSERT(inclusive_plan.pass_count == 2u);
  TEST_ASSERT(inclusive_plan.temp_bytes == 32u);
  return 0;
}

} // namespace program_compute_contract
