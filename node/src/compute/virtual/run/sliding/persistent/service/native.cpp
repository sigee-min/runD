#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

Status terminal_persistent_native(
    SlidingProductRun &state, SlidingProductWork &work,
    const node::accel::detail::PersistentResidencySlidingDoneObservation
        &observation) noexcept {
  const Status terminal = status_from(observation.check);
  const residency::execution::TerminalKind kind =
      terminal_from(observation.terminal);
  auto sliding = state.pool->authority().sliding();
  if (!work.native ||
      !sliding.terminal_execution_sliding_native(
          state.sliding, work.native, terminal, kind, observation.may_write) ||
      !sliding.release_execution_sliding_native(
          state.sliding, std::move(work.native))) {
    state.poison.store(true, std::memory_order_release);
    return Status::fail(Reason::CompletionInvalid);
  }
  work.terminaled = true;
  work.native_released = true;
  work.native_status = terminal;
  work.native_terminal = kind;
  work.native_may_write = observation.may_write;
  if (!terminal) {
    record_failure(state, terminal, observation.identity.coordinate);
  }
  return terminal;
}

} // namespace rund::compute::detail::sliding_product_detail
