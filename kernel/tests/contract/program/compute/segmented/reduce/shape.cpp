#include "local.hpp"

namespace program_compute_contract {

int SegmentedReduceShape() {
  const rund::kernel::SegmentedReducePlan u32_plan =
      rund::kernel::PlanSegmentedReduce(U32SegmentedReduce());
  TEST_ASSERT(u32_plan.ok);
  TEST_ASSERT(std::string_view{u32_plan.reason} == "ok");
  TEST_ASSERT(u32_plan.op == rund::kernel::ReduceOp::Sum);
  TEST_ASSERT(u32_plan.element == rund::kernel::ReduceElement::U32);
  TEST_ASSERT(u32_plan.element_count == 8u);
  TEST_ASSERT(u32_plan.element_bytes == 4u);
  TEST_ASSERT(u32_plan.head_bytes == 4u);
  TEST_ASSERT(u32_plan.block_size == 4u);
  TEST_ASSERT(u32_plan.block_count == 2u);
  TEST_ASSERT(u32_plan.pass_count == 2u);
  TEST_ASSERT(u32_plan.temp_value_bytes == 32u);
  TEST_ASSERT(u32_plan.temp_head_bytes == 32u);
  TEST_ASSERT(u32_plan.status_bytes == 4u);
  TEST_ASSERT(u32_plan.temp_bytes == 68u);
  TEST_ASSERT(rund::kernel::SegmentedReducePlanMatchesDesc(U32SegmentedReduce(),
                                                           u32_plan));
  rund::kernel::SegmentedReducePlan forged = u32_plan;
  ++forged.status_bytes;
  TEST_ASSERT(!rund::kernel::SegmentedReducePlanMatchesDesc(
      U32SegmentedReduce(), forged));

  rund::kernel::SegmentedReduceDesc u64_desc = U32SegmentedReduce();
  u64_desc.element = rund::kernel::ReduceElement::U64;
  u64_desc.element_count = 9u;
  const rund::kernel::SegmentedReducePlan u64_plan =
      rund::kernel::PlanSegmentedReduce(u64_desc);
  TEST_ASSERT(u64_plan.ok);
  TEST_ASSERT(u64_plan.element_bytes == 8u);
  TEST_ASSERT(u64_plan.temp_value_bytes == 72u);
  TEST_ASSERT(u64_plan.temp_head_bytes == 36u);
  TEST_ASSERT(u64_plan.temp_bytes == 112u);
  return 0;
}

} // namespace program_compute_contract
