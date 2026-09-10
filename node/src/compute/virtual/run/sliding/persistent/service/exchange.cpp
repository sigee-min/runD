#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {
namespace {

[[nodiscard]] PersistentServiceOutcome
abort_exchange(SlidingProductRun &state,
               const PersistentServiceCoordinate &service,
               const Status failure) noexcept {
  const node::accel::detail::PersistentResidencySlidingDoneObservation
      observation{
          .identity = service.identity,
          .check = persistent_service_check(failure),
          .terminal = node::accel::detail::NativeTerminal::UnknownMayWrite,
          .may_write = true,
      };
  const Status terminal =
      terminal_persistent_native(state, *service.work, observation);
  const Status reported = report_persistent_service_failure(
      state, service.identity, failure,
      node::accel::detail::NativeTerminal::UnknownMayWrite, true);
  return make_persistent_unknown_failure(
      state, terminal.reason() == failure.reason() &&
                     reported.reason() == failure.reason()
                 ? failure
                 : Status::fail(Reason::CompletionInvalid));
}

[[nodiscard]] PersistentServiceOutcome finish_failed_exchange(
    SlidingProductRun &state, const PersistentServiceCoordinate &service,
    const node::accel::detail::PersistentResidencySlidingDoneObservation
        &observation,
    const Status failure) noexcept {
  if (observation.terminal ==
      node::accel::detail::NativeTerminal::UnknownMayWrite) {
    return make_persistent_unknown_failure(state, failure);
  }
  const Status reported = report_persistent_service_failure(
      state, service.identity, failure,
      node::accel::detail::NativeTerminal::Known, observation.may_write);
  const Status acknowledged =
      acknowledge_persistent_service(state, service.identity, failure);
  return reported.reason() == failure.reason() &&
                 acknowledged.reason() == failure.reason()
             ? make_persistent_known_failure(failure)
             : make_persistent_unknown_failure(
                   state, Status::fail(Reason::CompletionInvalid));
}

} // namespace

PersistentServiceOutcome exchange_persistent_service_native(
    SlidingProductRun &state, PersistentServiceCoordinate &service) noexcept {
  const auto &operations = state.persistent_preparation.backend.service;
  if (operations.signal_ready == nullptr) {
    record_service_fault(state, ServiceFaultStage::Ops, 1u, service.coordinate,
                         Status::fail(Reason::DeviceLost), 1u);
    return abort_exchange(state, service, Status::fail(Reason::DeviceLost));
  }
  if (operations.wait_done == nullptr) {
    record_service_fault(state, ServiceFaultStage::Ops, 2u, service.coordinate,
                         Status::fail(Reason::DeviceLost), 2u);
    return abort_exchange(state, service, Status::fail(Reason::DeviceLost));
  }
  if (!operations
           .signal_ready(
               state.persistent_control,
               node::accel::detail::PersistentResidencySlidingReadySignal{
                   .identity = service.identity,
                   .admission = persistent_service_check(Status::success()),
               })
           .ok) {
    record_service_fault(state, ServiceFaultStage::Signal, 1u,
                         service.coordinate, Status::fail(Reason::DeviceLost),
                         1u);
    return abort_exchange(state, service, Status::fail(Reason::DeviceLost));
  }
  node::accel::detail::PersistentResidencySlidingDoneObservation observation{};
  if (!operations
           .wait_done(state.persistent_control,
                      node::accel::detail::PersistentResidencySlidingDoneWait{
                          .identity = service.identity},
                      observation)
           .ok) {
    record_service_fault(state, ServiceFaultStage::Wait, 1u, service.coordinate,
                         Status::fail(Reason::DeviceLost), 1u);
    return abort_exchange(state, service, Status::fail(Reason::DeviceLost));
  }
  if (!node::accel::detail::persistent_sliding_same_service_identity(
          service.identity, observation.identity)) {
    record_service_fault(state, ServiceFaultStage::Wait, 2u, service.coordinate,
                         Status::fail(Reason::DeviceLost), 2u);
    return abort_exchange(state, service, Status::fail(Reason::DeviceLost));
  }
  const Status native =
      terminal_persistent_native(state, *service.work, observation);
  if (!native) {
    return finish_failed_exchange(state, service, observation, native);
  }
  state.pipeline_submitted[service.coordinate % 2u] = observation.dispatched;
  return make_persistent_completed();
}

} // namespace rund::compute::detail::sliding_product_detail
