#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

void quarantine_persistent_ticket_locked(SlidingProductRun &) noexcept;

void final_persistent(
    void *const raw,
    node::accel::detail::PersistentResidencySlidingFinal &&final) noexcept {
  auto *const state = static_cast<SlidingProductRun *>(raw);
  if (state == nullptr) {
    return;
  }
  std::lock_guard lock{state->gate};
  if (state->persistent_final_received ||
      !node::accel::detail::persistent_sliding_final_valid(
          state->persistent_preparation.active_request(), final)) {
    quarantine_persistent_ticket_locked(*state);
    state->poison.store(true, std::memory_order_release);
    state->failure = Status::fail(Reason::CompletionInvalid);
    state->failure_coordinate =
        node::accel::detail::PersistentSlidingNoCoordinate;
  } else if (final.terminal ==
             node::accel::detail::NativeTerminal::UnknownMayWrite) {
    quarantine_persistent_ticket_locked(*state);
    state->persistent_final = std::move(final);
    state->persistent_final_received = true;
  } else {
    state->persistent_final = std::move(final);
    state->persistent_final_received = true;
  }
  state->ready.notify_all();
}

} // namespace rund::compute::detail::sliding_product_detail
