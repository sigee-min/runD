#include "local.hpp"

namespace program_compute_contract {

int SegmentedReduceReject() {
  rund::kernel::SegmentedReduceDesc zero_count = U32SegmentedReduce();
  zero_count.element_count = 0u;
  const rund::kernel::SegmentedReducePlan zero_count_plan =
      rund::kernel::PlanSegmentedReduce(zero_count);
  TEST_ASSERT(!zero_count_plan.ok);
  TEST_ASSERT(std::string_view{zero_count_plan.reason} ==
              "compute_segmented_reduce_count_zero");

  rund::kernel::SegmentedReduceDesc unknown_op = U32SegmentedReduce();
  unknown_op.op = static_cast<rund::kernel::ReduceOp>(0u);
  const rund::kernel::SegmentedReducePlan unknown_op_plan =
      rund::kernel::PlanSegmentedReduce(unknown_op);
  TEST_ASSERT(!unknown_op_plan.ok);
  TEST_ASSERT(std::string_view{unknown_op_plan.reason} ==
              "compute_segmented_reduce_op_unsupported");

  rund::kernel::SegmentedReduceDesc unknown_element = U32SegmentedReduce();
  unknown_element.element = static_cast<rund::kernel::ReduceElement>(0u);
  const rund::kernel::SegmentedReducePlan unknown_element_plan =
      rund::kernel::PlanSegmentedReduce(unknown_element);
  TEST_ASSERT(!unknown_element_plan.ok);
  TEST_ASSERT(std::string_view{unknown_element_plan.reason} ==
              "compute_segmented_reduce_element_unsupported");

  rund::kernel::SegmentedReduceDesc zero_block = U32SegmentedReduce();
  zero_block.block_size = 0u;
  const rund::kernel::SegmentedReducePlan zero_block_plan =
      rund::kernel::PlanSegmentedReduce(zero_block);
  TEST_ASSERT(!zero_block_plan.ok);
  TEST_ASSERT(std::string_view{zero_block_plan.reason} ==
              "compute_segmented_reduce_block_invalid");

  rund::kernel::SegmentedReduceDesc overflow = U32SegmentedReduce();
  overflow.element_count =
      (std::numeric_limits<rund::kernel::u64>::max() / 8u) + 1u;
  const rund::kernel::SegmentedReducePlan overflow_plan =
      rund::kernel::PlanSegmentedReduce(overflow);
  TEST_ASSERT(!overflow_plan.ok);
  TEST_ASSERT(std::string_view{overflow_plan.reason} ==
              "compute_segmented_reduce_temp_overflow");
  return 0;
}

}  // namespace program_compute_contract
