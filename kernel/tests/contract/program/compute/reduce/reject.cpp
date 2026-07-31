#include "local.hpp"

namespace program_compute_contract {

int ReduceReject() {
  rund::kernel::ReduceDesc unknown_op = U32Reduce();
  unknown_op.op = static_cast<rund::kernel::ReduceOp>(0u);
  const rund::kernel::ReducePlan unknown_op_plan =
      rund::kernel::PlanReduce(unknown_op);
  TEST_ASSERT(!unknown_op_plan.ok);
  TEST_ASSERT(std::string_view{unknown_op_plan.reason} ==
              "compute_reduce_op_unsupported");

  rund::kernel::ReduceDesc unknown_count_source = U32Reduce();
  unknown_count_source.count_source =
      static_cast<rund::kernel::ComputeCountSource>(255u);
  const rund::kernel::ReducePlan unknown_count_source_plan =
      rund::kernel::PlanReduce(unknown_count_source);
  TEST_ASSERT(!unknown_count_source_plan.ok);
  TEST_ASSERT(std::string_view{unknown_count_source_plan.reason} ==
              "compute_reduce_count_source_unsupported");

  rund::kernel::ReduceDesc unknown_element = U32Reduce();
  unknown_element.element = static_cast<rund::kernel::ReduceElement>(0u);
  const rund::kernel::ReducePlan unknown_element_plan =
      rund::kernel::PlanReduce(unknown_element);
  TEST_ASSERT(!unknown_element_plan.ok);
  TEST_ASSERT(std::string_view{unknown_element_plan.reason} ==
              "compute_reduce_element_unsupported");

  rund::kernel::ReduceDesc zero_count = U32Reduce();
  zero_count.element_count = 0u;
  const rund::kernel::ReducePlan zero_count_plan =
      rund::kernel::PlanReduce(zero_count);
  TEST_ASSERT(!zero_count_plan.ok);
  TEST_ASSERT(std::string_view{zero_count_plan.reason} ==
              "compute_reduce_count_zero");

  rund::kernel::ReduceDesc zero_block = U32Reduce();
  zero_block.block_size = 0u;
  const rund::kernel::ReducePlan zero_block_plan =
      rund::kernel::PlanReduce(zero_block);
  TEST_ASSERT(!zero_block_plan.ok);
  TEST_ASSERT(std::string_view{zero_block_plan.reason} ==
              "compute_reduce_block_invalid");

  rund::kernel::ReduceDesc overflow = U32Reduce();
  overflow.op = rund::kernel::ReduceOp::Min;
  overflow.element = rund::kernel::ReduceElement::U64;
  overflow.element_count = std::numeric_limits<rund::kernel::u64>::max();
  overflow.block_size = 2u;
  const rund::kernel::ReducePlan overflow_plan =
      rund::kernel::PlanReduce(overflow);
  TEST_ASSERT(!overflow_plan.ok);
  TEST_ASSERT(std::string_view{overflow_plan.reason} ==
              "compute_reduce_temp_overflow");
  return 0;
}

} // namespace program_compute_contract
