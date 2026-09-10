#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

Status suppress_persistent_service_coordinate(SlidingProductRun &state,
                                              const std::uint64_t coordinate,
                                              const Status failure) noexcept {
  node::accel::detail::PersistentResidencySlidingServiceIdentity identity{};
  if (failure || !node::accel::detail::persistent_sliding_service_identity(
                     state.persistent_preparation.active_request(), coordinate,
                     identity)) {
    state.poison.store(true, std::memory_order_release);
    return Status::fail(Reason::CompletionInvalid);
  }
  const auto &service = state.persistent_preparation.backend.service;
  if (service.signal_ready == nullptr || service.wait_done == nullptr ||
      !service
           .signal_ready(
               state.persistent_control,
               node::accel::detail::PersistentResidencySlidingReadySignal{
                   .identity = identity,
                   .admission = persistent_service_check(failure),
               })
           .ok) {
    state.poison.store(true, std::memory_order_release);
    return Status::fail(Reason::CompletionInvalid);
  }
  node::accel::detail::PersistentResidencySlidingDoneObservation observation{};
  if (!service
           .wait_done(state.persistent_control,
                      node::accel::detail::PersistentResidencySlidingDoneWait{
                          .identity = identity},
                      observation)
           .ok ||
      !node::accel::detail::persistent_sliding_same_service_identity(
          identity, observation.identity) ||
      observation.check.ok ||
      observation.terminal != node::accel::detail::NativeTerminal::Known ||
      observation.dispatched || !observation.completed ||
      observation.may_write) {
    state.poison.store(true, std::memory_order_release);
    return Status::fail(Reason::CompletionInvalid);
  }
  const Status acknowledged =
      acknowledge_persistent_service(state, identity, failure);
  return acknowledged ? Status::fail(Reason::CompletionInvalid) : failure;
}

Status service_persistent_known_suffix(SlidingProductRun &state,
                                       const std::uint64_t first,
                                       const Status failure) noexcept {
  std::uint64_t end = 0u;
  {
    std::lock_guard lock{state.persistent_control.gate};
    end = node::accel::detail::persistent_sliding_accepted_end(
        state.persistent_control);
  }
  for (std::uint64_t coordinate = first;
       coordinate < end; ++coordinate) {
    const Status suppressed =
        suppress_persistent_service_coordinate(state, coordinate, failure);
    if (suppressed.reason() != failure.reason()) {
      return suppressed;
    }
  }
  std::lock_guard lock{state.gate};
  return state.persistent_final_received
             ? failure
             : Status::fail(Reason::CompletionInvalid);
}

} // namespace rund::compute::detail::sliding_product_detail
