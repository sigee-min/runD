#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

void quarantine_persistent_ticket_locked(SlidingProductRun &state) noexcept {
  const auto request = state.persistent_preparation.active_request();
  node::accel::detail::persistent_sliding_quarantine_ticket(request.ticket);
}

void quarantine_final(SlidingProductRun &state) noexcept {
  bool paired = false;
  if (state.pool != nullptr && state.cold != nullptr) {
    paired = state.pool->authority().sliding().quarantine_bound_execution_sliding(
        state.cold->plan, state.sliding, Status::fail(Reason::DeviceLost));
  }
  if (!paired) {
    state.poison.store(true, std::memory_order_release);
  }
  std::lock_guard lock{state.gate};
  quarantine_persistent_ticket_locked(state);
  state.result.status = Status::fail(Reason::DeviceLost);
  state.result.poison_pipeline = true;
  state.quarantine = state.shared_from_this();
  state.completed = true;
  state.ready.notify_all();
}

} // namespace rund::compute::detail::sliding_product_detail
