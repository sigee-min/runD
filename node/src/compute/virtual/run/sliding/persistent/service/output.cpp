#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

PersistentServiceOutcome service_persistent_coordinate_output(
    SlidingProductRun &state, PersistentServiceCoordinate &service) noexcept {
  Status output = Status::success();
  const OutputServiceResult serviced =
      service_output(state, *service.work, output);
  if (serviced == OutputServiceResult::Pending) {
    return make_persistent_unknown_failure(state,
                                           Status::fail(Reason::PipelineBusy));
  }
  if (serviced == OutputServiceResult::Failed) {
    record_failure(state, output, service.coordinate);
    const Status reported = report_persistent_service_failure(
        state, service.identity, output,
        node::accel::detail::NativeTerminal::Known, false);
    const Status acknowledged =
        acknowledge_persistent_service(state, service.identity, output);
    return reported.reason() == output.reason() &&
                   acknowledged.reason() == output.reason()
               ? make_persistent_known_failure(output)
               : make_persistent_unknown_failure(
                     state, Status::fail(Reason::CompletionInvalid));
  }
  const Status acknowledged = acknowledge_persistent_service(
      state, service.identity, Status::success());
  return acknowledged ? make_persistent_completed()
                      : make_persistent_unknown_failure(state, acknowledged);
}

} // namespace rund::compute::detail::sliding_product_detail
