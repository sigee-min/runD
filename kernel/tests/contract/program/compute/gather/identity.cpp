#include "local.hpp"

namespace program_compute_contract {

int GatherIdentity() {
  const rund::kernel::GatherDesc desc = U32Gather();
  const rund::kernel::GatherHash first =
      rund::kernel::HashGather(desc);
  const rund::kernel::GatherHash second =
      rund::kernel::HashGather(desc);
  rund::kernel::GatherDesc changed = desc;
  changed.source_count = 17u;
  const rund::kernel::GatherHash changed_hash =
      rund::kernel::HashGather(changed);
  changed = desc;
  changed.count_source = rund::kernel::ComputeCountSource::BufferU32;
  const rund::kernel::GatherHash bounded_hash =
      rund::kernel::HashGather(changed);

  TEST_ASSERT(first.hi == second.hi);
  TEST_ASSERT(first.lo == second.lo);
  TEST_ASSERT(first.hi != 0u || first.lo != 0u);
  TEST_ASSERT(first.hi != changed_hash.hi || first.lo != changed_hash.lo);
  TEST_ASSERT(first.hi != bounded_hash.hi || first.lo != bounded_hash.lo);
  return 0;
}

} // namespace program_compute_contract
