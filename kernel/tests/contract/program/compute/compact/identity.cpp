#include "local.hpp"

namespace program_compute_contract {

int CompactIdentity() {
  const rund::kernel::CompactPlan first =
      rund::kernel::PlanCompact(U32Compact());
  const rund::kernel::CompactPlan second =
      rund::kernel::PlanCompact(U32Compact());
  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(SameCompactPlan(first, second));

  const rund::kernel::CompactDesc desc = U32Compact();
  const rund::kernel::CompactHash first_hash =
      rund::kernel::HashCompact(desc);
  const rund::kernel::CompactHash second_hash =
      rund::kernel::HashCompact(desc);
  rund::kernel::CompactDesc changed = desc;
  changed.output_capacity -= 1u;
  const rund::kernel::CompactHash changed_hash =
      rund::kernel::HashCompact(changed);

  TEST_ASSERT(first_hash.hi == second_hash.hi);
  TEST_ASSERT(first_hash.lo == second_hash.lo);
  TEST_ASSERT(first_hash.hi != 0u || first_hash.lo != 0u);
  TEST_ASSERT(first_hash.hi != changed_hash.hi ||
              first_hash.lo != changed_hash.lo);
  return 0;
}

} // namespace program_compute_contract
