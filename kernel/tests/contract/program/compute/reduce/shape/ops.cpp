#include "../local.hpp"

namespace program_compute_contract {

int ReduceShapeOps() {
  const rund::kernel::ReducePlan sum =
      rund::kernel::PlanReduce(U32Reduce());
  const rund::kernel::ReducePlan min =
      rund::kernel::PlanReduce(ReduceWithOp(rund::kernel::ReduceOp::Min));
  TEST_ASSERT(min.ok);
  TEST_ASSERT(min.op == rund::kernel::ReduceOp::Min);
  TEST_ASSERT(min.items_per_thread == 1u);
  TEST_ASSERT(min.first_pass_group_count == 5u);
  TEST_ASSERT(min.partial_element_bytes == sizeof(rund::kernel::u32));
  TEST_ASSERT(min.partial_bytes == 20u);
  TEST_ASSERT(min.temp_bytes == 24u);

  const rund::kernel::ReducePlan max =
      rund::kernel::PlanReduce(ReduceWithOp(rund::kernel::ReduceOp::Max));
  TEST_ASSERT(max.ok);
  TEST_ASSERT(max.op == rund::kernel::ReduceOp::Max);
  TEST_ASSERT(max.pass_count == 2u);

  rund::kernel::ReduceDesc bounded_desc = U32Reduce();
  bounded_desc.count_source = rund::kernel::ComputeCountSource::BufferU32;
  const rund::kernel::ReducePlan bounded =
      rund::kernel::PlanReduce(bounded_desc);
  TEST_ASSERT(bounded.ok);
  TEST_ASSERT(bounded.element_count == sum.element_count);
  TEST_ASSERT(bounded.temp_bytes == sum.temp_bytes);
  TEST_ASSERT(bounded.count_source ==
              rund::kernel::ComputeCountSource::BufferU32);
  return 0;
}

}  // namespace program_compute_contract
