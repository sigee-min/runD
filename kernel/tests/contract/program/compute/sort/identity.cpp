#include "local.hpp"

namespace program_compute_contract {

int test_compute_sort_descriptor_hash_is_deterministic_and_field_sensitive() {
  const rund::kernel::SortDesc desc = U32Sort();
  const rund::kernel::SortHash first =
      rund::kernel::HashSort(desc);
  const rund::kernel::SortHash second =
      rund::kernel::HashSort(desc);
  rund::kernel::SortDesc changed = desc;
  changed.element_count += 1u;
  const rund::kernel::SortHash changed_hash =
      rund::kernel::HashSort(changed);
  rund::kernel::SortDesc narrowed = desc;
  narrowed.key_bits = 16u;
  const rund::kernel::SortHash narrowed_hash =
      rund::kernel::HashSort(narrowed);
  rund::kernel::SortDesc bounded = desc;
  bounded.count_source = rund::kernel::ComputeCountSource::BufferU32;
  const rund::kernel::SortHash bounded_hash =
      rund::kernel::HashSort(bounded);

  TEST_ASSERT(first.hi == second.hi);
  TEST_ASSERT(first.lo == second.lo);
  TEST_ASSERT(first.hi != 0u || first.lo != 0u);
  TEST_ASSERT(first.hi != changed_hash.hi || first.lo != changed_hash.lo);
  TEST_ASSERT(first.hi != narrowed_hash.hi || first.lo != narrowed_hash.lo);
  TEST_ASSERT(first.hi != bounded_hash.hi || first.lo != bounded_hash.lo);
  return 0;
}

int test_compute_sort_plan_is_deterministic() {
  const rund::kernel::SortPlan first = rund::kernel::PlanSort(U32Sort());
  const rund::kernel::SortPlan second =
      rund::kernel::PlanSort(U32Sort());

  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(SamePlan(first, second));
  return 0;
}

}  // namespace program_compute_contract
