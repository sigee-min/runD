#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {
namespace {

[[nodiscard]] PersistentServiceOutcome
reject_input(SlidingProductRun &state,
             const PersistentServiceCoordinate &service,
             const Status failure) noexcept {
  record_failure(state, failure, service.coordinate);
  if (!state.pool->authority().sliding().reject_execution_sliding_coordinate(
          state.cold->plan, state.sliding, service.coordinate, failure)) {
    return make_persistent_unknown_failure(
        state, Status::fail(Reason::CompletionInvalid));
  }
  const Status suppressed = suppress_persistent_service_coordinate(
      state, service.coordinate, failure);
  if (suppressed.reason() != failure.reason()) {
    return make_persistent_unknown_failure(state, suppressed);
  }
  if (!discard_unissued_projection(*service.work)) {
    return make_persistent_unknown_failure(
        state, Status::fail(Reason::CompletionInvalid));
  }
  return make_persistent_known_failure(failure);
}

} // namespace

PersistentServiceOutcome
admit_persistent_service_input(SlidingProductRun &state,
                               PersistentServiceCoordinate &service) noexcept {
  Status admission = Status::success();
  const InputServiceResult fetched =
      service_fetches(state, *service.work, admission);
  if (fetched == InputServiceResult::Pending) {
    return make_persistent_unknown_failure(
        state, Status::fail(Reason::CompletionInvalid));
  }
  if (fetched == InputServiceResult::Failed) {
    return reject_input(state, service, admission);
  }
  const InputServiceResult promoted =
      service_promote(state, *service.work, service.coordinate, admission);
  if (promoted == InputServiceResult::Pending) {
    return make_persistent_unknown_failure(
        state, Status::fail(Reason::CompletionInvalid));
  }
  return promoted == InputServiceResult::Failed
             ? reject_input(state, service, admission)
             : make_persistent_completed();
}

} // namespace rund::compute::detail::sliding_product_detail
