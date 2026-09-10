#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

rund::AccelCheck persistent_service_check(const Status status) noexcept {
  return status
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, reason_message(status.reason()).data()};
}

Status report_persistent_service_failure(
    SlidingProductRun &state,
    const node::accel::detail::PersistentResidencySlidingServiceIdentity
        &identity,
    const Status failure, const node::accel::detail::NativeTerminal terminal,
    const bool may_write) noexcept {
  const auto operation =
      state.persistent_preparation.backend.service.fail_service;
  if (operation == nullptr ||
      !operation(state.persistent_control,
                 node::accel::detail::PersistentResidencySlidingServiceFailure{
                     .identity = identity,
                     .check = persistent_service_check(failure),
                     .terminal = terminal,
                     .may_write = may_write,
                 })
           .ok) {
    state.poison.store(true, std::memory_order_release);
    return Status::fail(Reason::CompletionInvalid);
  }
  if (terminal == node::accel::detail::NativeTerminal::UnknownMayWrite) {
    state.poison.store(true, std::memory_order_release);
  }
  return failure;
}

Status acknowledge_persistent_service(
    SlidingProductRun &state,
    const node::accel::detail::PersistentResidencySlidingServiceIdentity
        &identity,
    const Status service_status) noexcept {
  const auto operation = state.persistent_preparation.backend.service.ack_done;
  if (operation == nullptr ||
      !operation(state.persistent_control,
                 node::accel::detail::PersistentResidencySlidingAcknowledgeDone{
                     .identity = identity,
                     .service_check = persistent_service_check(service_status),
                 })
           .ok) {
    state.poison.store(true, std::memory_order_release);
    return Status::fail(Reason::CompletionInvalid);
  }
  return service_status;
}

} // namespace rund::compute::detail::sliding_product_detail
