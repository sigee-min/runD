#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

void complete_known_final(SlidingProductRun &state,
                          const Status result) noexcept {
  {
    std::lock_guard lock{state.gate};
    state.result.status = result;
    state.result.output_hash = result ? state.hash.Finish() : 0u;
    state.result.poison_pipeline = state.poison.load(std::memory_order_acquire);
    state.completed = true;
  }
  state.ready.notify_all();
}

} // namespace rund::compute::detail::sliding_product_detail
