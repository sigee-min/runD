#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

bool inject_persistent_unknown_failure(
    SlidingProductRun &state, const PersistentServiceCoordinate &service,
    PersistentServiceOutcome &outcome) noexcept {
  if (!consume_persistent_service_unknown_injection()) {
    return false;
  }
  const Status failure = Status::fail(Reason::DeviceLost);
  record_failure(state, failure, service.coordinate);
  if (!state.pool->authority().sliding().quarantine_execution_sliding_coordinate(
          state.cold->plan, state.sliding, service.coordinate, failure)) {
    outcome = make_persistent_unknown_failure(
        state, Status::fail(Reason::CompletionInvalid));
    return true;
  }
  const Status reported = report_persistent_service_failure(
      state, service.identity, failure,
      node::accel::detail::NativeTerminal::UnknownMayWrite, true);
  outcome = make_persistent_unknown_failure(
      state, reported.reason() == failure.reason() ? failure : reported);
  return true;
}

} // namespace rund::compute::detail::sliding_product_detail
