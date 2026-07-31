#include "local.hpp"

namespace program_compute_contract {

int SegmentedReduceIdentity() {
  const rund::kernel::SegmentedReducePlan first =
      rund::kernel::PlanSegmentedReduce(U32SegmentedReduce());
  const rund::kernel::SegmentedReducePlan second =
      rund::kernel::PlanSegmentedReduce(U32SegmentedReduce());
  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(first.element_count == second.element_count);
  TEST_ASSERT(first.element_bytes == second.element_bytes);
  TEST_ASSERT(first.block_size == second.block_size);
  TEST_ASSERT(first.block_count == second.block_count);
  TEST_ASSERT(first.pass_count == second.pass_count);
  TEST_ASSERT(first.temp_value_bytes == second.temp_value_bytes);
  TEST_ASSERT(first.temp_head_bytes == second.temp_head_bytes);
  TEST_ASSERT(first.temp_bytes == second.temp_bytes);

  const rund::kernel::SegmentedReduceDesc desc = U32SegmentedReduce();
  const rund::kernel::SegmentedReduceHash first_hash =
      rund::kernel::HashSegmentedReduce(desc);
  const rund::kernel::SegmentedReduceHash second_hash =
      rund::kernel::HashSegmentedReduce(desc);
  rund::kernel::SegmentedReduceDesc changed = desc;
  changed.op = rund::kernel::ReduceOp::Max;
  const rund::kernel::SegmentedReduceHash changed_hash =
      rund::kernel::HashSegmentedReduce(changed);

  TEST_ASSERT(first_hash.hi == second_hash.hi);
  TEST_ASSERT(first_hash.lo == second_hash.lo);
  TEST_ASSERT(first_hash.hi != 0u || first_hash.lo != 0u);
  TEST_ASSERT(first_hash.hi != changed_hash.hi ||
              first_hash.lo != changed_hash.lo);
  return 0;
}

}  // namespace program_compute_contract
