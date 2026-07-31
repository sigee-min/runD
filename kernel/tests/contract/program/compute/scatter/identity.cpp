#include "local.hpp"

namespace program_compute_contract {

int ScatterIdentity() {
  const rund::kernel::ScatterDesc desc = U32Scatter();
  const rund::kernel::ScatterHash first =
      rund::kernel::HashScatter(desc);
  const rund::kernel::ScatterHash second =
      rund::kernel::HashScatter(desc);
  rund::kernel::ScatterDesc changed = desc;
  changed.output_count = 9u;
  const rund::kernel::ScatterHash changed_hash =
      rund::kernel::HashScatter(changed);

  TEST_ASSERT(first.hi == second.hi);
  TEST_ASSERT(first.lo == second.lo);
  TEST_ASSERT(first.hi != 0u || first.lo != 0u);
  TEST_ASSERT(first.hi != changed_hash.hi || first.lo != changed_hash.lo);
  return 0;
}

} // namespace program_compute_contract
