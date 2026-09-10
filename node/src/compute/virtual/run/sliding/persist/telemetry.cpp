#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

void record_persist_io(SlidingProductRun &state, const PersistIo &io) noexcept {
  std::lock_guard lock{state.gate};
  state.backing_io_ns += pipeline_clock() - io.started_ns;
  state.backing_write_bytes += io.bytes;
  if (io.status) {
    state.hash.Bytes(
        reinterpret_cast<const std::uint8_t *>(io.frame + io.hash_offset),
        static_cast<std::size_t>(io.bytes));
  }
  state.ready.notify_all();
}

} // namespace rund::compute::detail::sliding_product_detail
