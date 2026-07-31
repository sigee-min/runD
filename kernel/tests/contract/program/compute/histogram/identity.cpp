#include "local.hpp"

namespace program_compute_contract {

int HistogramIdentity() {
  const rund::kernel::HistogramDesc desc = U32Histogram();
  const rund::kernel::HistogramHash a =
      rund::kernel::HashHistogram(desc);
  const rund::kernel::HistogramHash b =
      rund::kernel::HashHistogram(desc);
  TEST_ASSERT(a.hi == b.hi);
  TEST_ASSERT(a.lo == b.lo);

  rund::kernel::HistogramDesc changed = desc;
  changed.element_count += 1u;
  const rund::kernel::HistogramHash c =
      rund::kernel::HashHistogram(changed);
  TEST_ASSERT(a.hi != c.hi || a.lo != c.lo);

  changed = desc;
  changed.bin_count += 1u;
  const rund::kernel::HistogramHash d =
      rund::kernel::HashHistogram(changed);
  TEST_ASSERT(a.hi != d.hi || a.lo != d.lo);
  return 0;
}

}  // namespace program_compute_contract
