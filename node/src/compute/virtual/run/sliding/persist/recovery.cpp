#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

void require_persist_recovery(SlidingProductRun &state) noexcept {
  if (state.recovery) {
    return;
  }
  VirtualBackingAccess::require_recovery(*state.output,
                                         state.run->active.output_bytes);
  state.recovery = true;
}

} // namespace rund::compute::detail::sliding_product_detail
