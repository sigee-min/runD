#include "../local.hpp"

namespace program_compute_contract {

int ReduceShape() {
  const rund::kernel::ReducePlan tree = rund::kernel::PlanReduce(U32Reduce());
  TEST_ASSERT(tree.ok);
  TEST_ASSERT(std::string_view{tree.reason} == "ok");
  TEST_ASSERT(tree.op == rund::kernel::ReduceOp::Sum);
  TEST_ASSERT(tree.element == rund::kernel::ReduceElement::U32);
  TEST_ASSERT(tree.element_count == 1025u);
  TEST_ASSERT(tree.element_bytes == 4u);
  TEST_ASSERT(tree.block_size == 256u);
  TEST_ASSERT(tree.items_per_thread == 8u);
  TEST_ASSERT(tree.first_pass_group_count == 1u);
  TEST_ASSERT(tree.pass_count == 1u);
  TEST_ASSERT(tree.partial_element_count == 0u);
  TEST_ASSERT(tree.partial_element_bytes == 16u);
  TEST_ASSERT(tree.partial_bytes == 0u);
  TEST_ASSERT(tree.status_bytes == 4u);
  TEST_ASSERT(tree.temp_bytes == 4u);
  TEST_ASSERT(tree.count_source ==
              rund::kernel::ComputeCountSource::Descriptor);
  TEST_ASSERT(rund::kernel::ReducePlanMatchesDesc(U32Reduce(), tree));
  rund::kernel::ReducePlan forged = tree;
  ++forged.partial_bytes;
  TEST_ASSERT(!rund::kernel::ReducePlanMatchesDesc(U32Reduce(), forged));

  rund::kernel::ReduceDesc single_desc = U32Reduce();
  single_desc.element = rund::kernel::ReduceElement::U64;
  single_desc.element_count = 1u;
  const rund::kernel::ReducePlan single = rund::kernel::PlanReduce(single_desc);
  TEST_ASSERT(single.ok);
  TEST_ASSERT(single.element_bytes == 8u);
  TEST_ASSERT(single.items_per_thread == 8u);
  TEST_ASSERT(single.first_pass_group_count == 1u);
  TEST_ASSERT(single.pass_count == 1u);
  TEST_ASSERT(single.partial_element_count == 0u);
  TEST_ASSERT(single.partial_element_bytes == 16u);
  TEST_ASSERT(single.partial_bytes == 0u);
  TEST_ASSERT(single.status_bytes == 4u);
  TEST_ASSERT(single.temp_bytes == 4u);

  const rund::kernel::ReducePlan count_tree = rund::kernel::PlanReduce(
      ReduceWithOp(rund::kernel::ReduceOp::CountNonzero));
  TEST_ASSERT(count_tree.ok);
  TEST_ASSERT(count_tree.op == rund::kernel::ReduceOp::CountNonzero);
  TEST_ASSERT(count_tree.element == rund::kernel::ReduceElement::U32);
  TEST_ASSERT(count_tree.pass_count == tree.pass_count);
  TEST_ASSERT(count_tree.partial_bytes == tree.partial_bytes);
  TEST_ASSERT(count_tree.temp_bytes == tree.temp_bytes);

  rund::kernel::ReduceDesc hierarchy_desc = U32Reduce();
  hierarchy_desc.element_count = 262144u;
  const rund::kernel::ReducePlan hierarchy =
      rund::kernel::PlanReduce(hierarchy_desc);
  TEST_ASSERT(hierarchy.ok);
  TEST_ASSERT(hierarchy.items_per_thread == 8u);
  TEST_ASSERT(hierarchy.first_pass_group_count == 128u);
  TEST_ASSERT(hierarchy.pass_count == 2u);
  TEST_ASSERT(hierarchy.partial_element_count == 128u);
  TEST_ASSERT(hierarchy.partial_element_bytes == 16u);
  TEST_ASSERT(hierarchy.partial_bytes == 2048u);
  TEST_ASSERT(hierarchy.temp_bytes == 2052u);

  hierarchy_desc.element_count = 262145u;
  const rund::kernel::ReducePlan capped =
      rund::kernel::PlanReduce(hierarchy_desc);
  TEST_ASSERT(capped.ok);
  TEST_ASSERT(capped.first_pass_group_count == 128u);
  TEST_ASSERT(capped.pass_count == 2u);
  TEST_ASSERT(capped.temp_bytes == hierarchy.temp_bytes);

  return 0;
}

} // namespace program_compute_contract
