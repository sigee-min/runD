#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {
namespace {

void saturating_add(std::uint64_t &target, const std::uint64_t value) noexcept {
  target = value > std::numeric_limits<std::uint64_t>::max() - target
               ? std::numeric_limits<std::uint64_t>::max()
               : target + value;
}

} // namespace

void fold_sliding_stats(
    SlidingProductRun &state,
    const residency::execution::SlidingEvidence &evidence,
    const node::accel::detail::BackendResidencySlidingFinal &native) noexcept {
  ResidencyStats &residency = state.stats->pipeline.residency;
  saturating_add(residency.epoch_count, evidence.admitted);
  saturating_add(residency.window_handoff_count, 1u);
  saturating_add(residency.window_batch_count, native.accepted_coordinates);
  saturating_add(residency.window_queue_call_count, native.queue_calls);
  // Fetch calls and cache hits are disjoint evidence counters: a call is a
  // backing miss, while a hit reuses an already resident Host row. Do not
  // subtract hits from the miss count; after a retained cross-run hit that
  // would underflow and corrupt the supplied-page total.
  saturating_add(residency.page_in_count, evidence.fetch_calls);
  saturating_add(residency.page_out_count, evidence.persist_calls);
  saturating_add(residency.backing_read_bytes, state.backing_read_bytes);
  saturating_add(residency.backing_write_bytes, state.backing_write_bytes);
  saturating_add(residency.backing_io_ns, state.backing_io_ns);
  saturating_add(residency.cache_hit_count, evidence.fetch_hits);
  saturating_add(residency.page_in_bytes, evidence.promote_bytes);
  saturating_add(residency.page_out_bytes, evidence.drain_bytes);
  residency.failed_page = state.result.failed_page;
}

} // namespace rund::compute::detail::sliding_product_detail
