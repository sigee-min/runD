#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

PersistentServiceOutcome
service_persistent_coordinate(SlidingProductRun &state,
                              const std::uint64_t coordinate) noexcept {
  PersistentServiceCoordinate service{};
  if (!initialize_persistent_service_coordinate(state, coordinate, service)) {
    return make_persistent_unknown_failure(
        state, Status::fail(Reason::CompletionInvalid));
  }
  PersistentServiceOutcome injected{};
  if (inject_persistent_unknown_failure(state, service, injected)) {
    return injected;
  }
  const PersistentServiceOutcome admitted =
      admit_persistent_service_input(state, service);
  if (admitted.disposition != PersistentServiceDisposition::Completed) {
    record_service_fault(state, ServiceFaultStage::Admit, 1u, coordinate,
                         admitted.status,
                         static_cast<std::uint64_t>(admitted.disposition));
    return admitted;
  }
  if (!select_persistent_service_native(state, service)) {
    record_service_fault(state, ServiceFaultStage::Select, 1u, coordinate,
                         Status::fail(Reason::CompletionInvalid), 1u);
    return make_persistent_unknown_failure(
        state, Status::fail(Reason::CompletionInvalid));
  }
  const PersistentServiceOutcome exchanged =
      exchange_persistent_service_native(state, service);
  return exchanged.disposition == PersistentServiceDisposition::Completed
             ? service_persistent_coordinate_output(state, service)
             : exchanged;
}

} // namespace rund::compute::detail::sliding_product_detail
