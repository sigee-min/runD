#include "internal.hpp"

#include <utility>

namespace rund::compute::detail::graph_reduce {

bool PrefetchController::retire(const std::size_t lane_index,
                                PrefetchLane &lane) noexcept {
  if (lane_index >= lanes_.size() || (!lane.pending && !lane.forecast)) {
    return false;
  }
  if (!lane.pending) {
    const residency::PrefetchReceipt receipt = pool_.prefetch[lane_index].wait();
    (void)receipt;
    const bool settled = settle(lane);
    if (settled) {
      lane = {};
    }
    return settled;
  }
  if (lane.device_only) {
    lane = {};
    return true;
  }

  const std::uint64_t started = pipeline_clock();
  const residency::PrefetchReceipt receipt = pool_.prefetch[lane_index].wait();
  const Interval wait{.started = started, .completed = pipeline_clock()};
  lane.pending = false;
  (void)record_interval(nullptr, wait, std::nullopt, stats_.pipeline.residency);
  observe(receipt);
  if (receipt.status) {
    std::uint64_t fetched = 0u;
    for (const residency::PrefetchedPage &page : receipt.pages) {
      fetched += static_cast<std::uint64_t>(page.fetched);
    }
    classify_backing(
        stats_,
        std::span<const residency::PageUse>{lane.sources.data(), lane.count},
        fetched, receipt.speculative, true);
  }
  static_cast<void>(terminal_graph_prefetch(authority_, lane.forecast, receipt));
  const bool settled = settle(lane);
  if (settled) {
    lane = {};
  }
  return settled;
}

} // namespace rund::compute::detail::graph_reduce
