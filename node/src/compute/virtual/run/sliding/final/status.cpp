#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

Status merged_run_status(SlidingProductRun &state, Status result) noexcept {
  std::lock_guard lock{state.gate};
  if (!state.failure) {
    result = state.failure;
  }
  return result;
}

} // namespace rund::compute::detail::sliding_product_detail
