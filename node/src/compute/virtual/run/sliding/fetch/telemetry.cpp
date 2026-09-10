#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

void record_fetch_io(SlidingProductRun &state, const FetchIo &io) noexcept {
  std::lock_guard lock{state.gate};
  state.backing_io_ns += pipeline_clock() - io.started_ns;
  state.backing_read_bytes += io.bytes;
}

} // namespace rund::compute::detail::sliding_product_detail
