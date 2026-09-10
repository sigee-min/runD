#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

void PrefetchController::observe(
    const residency::PrefetchReceipt &receipt) noexcept {
  using prefetch_detail::Accumulate;
  Accumulate(stats_.pipeline.residency.backing_io_ns, receipt.io_ns);
  if (!receipt.status) {
    return;
  }
  for (const residency::PrefetchedPage &page : receipt.pages) {
    if (page.fetched) {
      Accumulate(stats_.pipeline.residency.backing_read_bytes,
                 page.backing_bytes);
    }
  }
}

} // namespace rund::compute::detail::graph_reduce
