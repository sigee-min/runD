#include "local.hpp"

#include "src/compute/virtual/run/device_vsm/model.hpp"

namespace rund::measure::compute::route_matrix {

::rund::compute::Status prepare_route_observed(
    ::rund::compute::detail::VirtualPipelineState &state,
    const ::rund::compute::detail::VirtualRunProjection &run,
    const ::rund::compute::detail::VirtualDeviceVsmRouteProof proof,
    ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared
        &prepared) noexcept {
  RouteObserverImpl *const observer = active_route_observer;
  if (observer == nullptr || observer->original == nullptr ||
      observer->original->virtual_execution
              .prepare_virtual_device_vsm_product == nullptr) {
    return ::rund::compute::Status::fail(
        ::rund::compute::Reason::BackendUnsupported);
  }
  if (observer->pending_prepared) {
    mark_lifecycle_failure(*observer);
    observer->pending_prepared = false;
  }
  ++observer->evidence.prepare_attempts;
  observe_proof(*observer, proof);
  const ::rund::compute::Status status =
      observer->original->virtual_execution.prepare_virtual_device_vsm_product(
          state, run, proof, prepared);
  observer->evidence.backend = observer->device == nullptr
                                   ? ComputeBackend::Unavailable
                                   : observer->device->backend;
  if (prepared.owner != nullptr) {
    observer->observe_owner(prepared);
  }
  if (status) {
    ++observer->evidence.prepared_runs;
  }
  if (!status || prepared.owner == nullptr) {
    mark_lifecycle_failure(*observer);
    return status;
  }
  if (observer->prepared_count >= CohortRuns) {
    mark_lifecycle_failure(*observer);
  } else {
    ++observer->prepared_count;
    observer->pending_prepared = true;
  }
  return status;
}

::rund::compute::detail::VirtualExecutionResult execute_route_observed(
    ::rund::compute::detail::VirtualPipelineState &state,
    const std::span<::rund::compute::VirtualBacking *const> inputs,
    ::rund::compute::VirtualBacking &output,
    const ::rund::compute::detail::VirtualRunProjection &run,
    const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared &prepared,
    ::rund::compute::Stats &stats) noexcept {
  RouteObserverImpl *const observer = active_route_observer;
  if (observer == nullptr || observer->original == nullptr ||
      observer->original->virtual_execution
              .execute_virtual_device_vsm_product == nullptr) {
    return ::rund::compute::detail::VirtualExecutionResult{
        .status = ::rund::compute::Status::fail(
            ::rund::compute::Reason::BackendUnsupported)};
  }
  observer->evidence.backend = observer->device == nullptr
                                   ? ComputeBackend::Unavailable
                                   : observer->device->backend;
  const ::rund::compute::detail::VirtualExecutionResult result =
      observer->original->virtual_execution.execute_virtual_device_vsm_product(
          state, inputs, output, run, prepared, stats);
  observer->observe(prepared, result);
  return result;
}

} // namespace rund::measure::compute::route_matrix
