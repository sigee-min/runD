#include "internal.hpp"

namespace rund_node_test_virtual::product::route_detail {

Status
ObserveSlidingPrepare(VirtualPipelineState &state,
                      const VirtualRunProjection &run,
                      VirtualExecutionSlidingPrepared &prepared) noexcept {
  CallbackLease lease;
  if (!begin_callback(pipeline_device(state), lease) ||
      lease.forward->virtual_execution.prepare_virtual_sliding_product ==
          nullptr) {
    return Status::fail(rund::compute::Reason::PipelineBusy);
  }
  const Status status =
      lease.forward->virtual_execution.prepare_virtual_sliding_product(
          state, run, prepared);
  record_sliding_prepare(lease.observation, status);
  return status;
}

VirtualExecutionResult
ObserveSlidingExecute(VirtualPipelineState &state, VirtualBacking &input,
                      VirtualBacking &output, const VirtualRunProjection &run,
                      const VirtualExecutionSlidingPrepared &prepared,
                      Stats &stats) noexcept {
  CallbackLease lease;
  if (!begin_callback(pipeline_device(state), lease) ||
      lease.forward->virtual_execution.execute_virtual_sliding_product ==
          nullptr) {
    return VirtualExecutionResult{
        .status = Status::fail(rund::compute::Reason::PipelineBusy)};
  }
  const VirtualExecutionResult result =
      lease.forward->virtual_execution.execute_virtual_sliding_product(
          state, input, output, run, prepared, stats);
  record_status(lease.observation,
                &ProductRouteObservation::sliding_execute_status,
                result.status);
  record_accepted(lease.observation, lease.observation->sliding_execute_called,
                  lease.observation->sliding_execute_accepted,
                  static_cast<bool>(result.status), OwnerPersistent);
  return result;
}

Status
ObserveDeviceVsmPrepare(VirtualPipelineState &state,
                        const VirtualRunProjection &run,
                        const VirtualDeviceVsmRouteProof proof,
                        VirtualExecutionDeviceVsmPrepared &prepared) noexcept {
  if (!proof_tamper_rejected(proof)) {
    return Status::fail(rund::compute::Reason::PipelineInvalid);
  }
  CallbackLease lease;
  if (!begin_callback(pipeline_device(state), lease) ||
      lease.forward->virtual_execution.prepare_virtual_device_vsm_product ==
          nullptr) {
    return Status::fail(rund::compute::Reason::PipelineBusy);
  }
  const Status status =
      lease.forward->virtual_execution.prepare_virtual_device_vsm_product(
          state, run, proof, prepared);
  record_device_vsm_prepare(lease.observation, status, prepared.reason);
  return status;
}

VirtualExecutionResult ObserveDeviceVsmExecute(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &run,
    const VirtualExecutionDeviceVsmPrepared &prepared, Stats &stats) noexcept {
  CallbackLease lease;
  if (!begin_callback(pipeline_device(state), lease) ||
      lease.forward->virtual_execution.execute_virtual_device_vsm_product ==
          nullptr) {
    return VirtualExecutionResult{
        .status = Status::fail(rund::compute::Reason::PipelineBusy)};
  }
  const VirtualExecutionResult result =
      lease.forward->virtual_execution.execute_virtual_device_vsm_product(
          state, inputs, output, run, prepared, stats);
  record_accepted(lease.observation,
                  lease.observation->device_vsm_execute_called,
                  lease.observation->device_vsm_execute_accepted,
                  static_cast<bool>(result.status), OwnerDeviceVsm);
  return result;
}

Status ObserveServiceFreeDirect(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    bool &selected) noexcept {
  DeviceState *const device = state == nullptr || state->device == nullptr
                                  ? nullptr
                                  : state->device.get();
  CallbackLease lease;
  if (!begin_callback(device, lease) ||
      lease.forward->virtual_execution.run_service_free_direct_product ==
          nullptr) {
    return Status::fail(rund::compute::Reason::PipelineBusy);
  }
  const Status status =
      lease.forward->virtual_execution.run_service_free_direct_product(
          state, selected);
  if (static_cast<bool>(status) && selected) {
    record_owner(lease.observation, OwnerServiceFreeDirect);
  }
  return status;
}

rund::AccelCheck ObserveResidencyPipeline(
    const DeviceState &device,
    const rund::node::accel::detail::PreparedKernelPipeline &prepared,
    const std::span<const std::uint32_t> locals, std::shared_ptr<void> lifetime,
    const rund::node::accel::detail::PreparedPipelineCompletion completion,
    void *const user) noexcept {
  CallbackLease lease;
  if (!begin_callback(const_cast<DeviceState *>(&device), lease) ||
      lease.forward->residency.submit_residency_pipeline == nullptr) {
    return {false, "route_observer_busy"};
  }
  const rund::AccelCheck result =
      lease.forward->residency.submit_residency_pipeline(
          device, prepared, locals, std::move(lifetime), completion, user);
  record_residency_failure(lease.observation, result, prepared);
  record_accepted(lease.observation,
                  lease.observation->submit_residency_pipeline_called,
                  lease.observation->submit_residency_pipeline_accepted,
                  result.ok, OwnerAccelRolling);
  return result;
}

Status ObserveResidencyWindow(
    const DeviceState &device,
    const rund::node::accel::detail::PreparedResidencyWindowRequest &request,
    rund::node::accel::detail::PreparedResidencyWindowControl
        &control) noexcept {
  CallbackLease lease;
  if (!begin_callback(const_cast<DeviceState *>(&device), lease) ||
      lease.forward->residency.submit_residency_window == nullptr) {
    return Status::fail(rund::compute::Reason::PipelineBusy);
  }
  const Status result = lease.forward->residency.submit_residency_window(
      device, request, control);
  record_accepted(lease.observation, lease.observation->window_submit_called,
                  lease.observation->window_submit_accepted,
                  static_cast<bool>(result), OwnerWindow);
  return result;
}

Status ObserveResidencyStreamWindow(
    const DeviceState &device,
    const rund::node::accel::detail::PreparedResidencyWindowRequest &request,
    rund::node::accel::detail::PreparedResidencyWindowControl &window,
    rund::node::accel::detail::PreparedResidencyStreamControl
        &stream) noexcept {
  CallbackLease lease;
  if (!begin_callback(const_cast<DeviceState *>(&device), lease) ||
      lease.forward->residency.submit_residency_stream_window == nullptr) {
    return Status::fail(rund::compute::Reason::PipelineBusy);
  }
  const Status result = lease.forward->residency.submit_residency_stream_window(
      device, request, window, stream);
  record_accepted(lease.observation,
                  lease.observation->submit_residency_stream_window_called,
                  lease.observation->submit_residency_stream_window_accepted,
                  static_cast<bool>(result), OwnerAccelRolling);
  return result;
}

Status ObserveResidencySchedule(
    const DeviceState &device,
    const rund::node::accel::detail::PreparedResidencyScheduleRequest &request,
    rund::node::accel::detail::PreparedResidencyScheduleControl &schedule,
    rund::node::accel::detail::PreparedResidencyStreamControl
        &stream) noexcept {
  CallbackLease lease;
  if (!begin_callback(const_cast<DeviceState *>(&device), lease) ||
      lease.forward->residency.submit_residency_schedule == nullptr) {
    return Status::fail(rund::compute::Reason::PipelineBusy);
  }
  const Status result = lease.forward->residency.submit_residency_schedule(
      device, request, schedule, stream);
  record_accepted(lease.observation,
                  lease.observation->submit_residency_schedule_called,
                  lease.observation->submit_residency_schedule_accepted,
                  static_cast<bool>(result), OwnerAccelRolling);
  return result;
}

Status ObservePrepareResidencySliding(
    const DeviceState &device,
    const std::span<
        const rund::node::accel::detail::PreparedResidencySlidingRole>
        roles,
    const rund::node::accel::detail::ResidencySlidingMemory memory,
    rund::node::accel::detail::PreparedResidencySlidingControl
        &control) noexcept {
  CallbackLease lease;
  if (!begin_callback(const_cast<DeviceState *>(&device), lease) ||
      lease.forward->residency.prepare_residency_sliding == nullptr) {
    return Status::fail(rund::compute::Reason::PipelineBusy);
  }
  const Status result = lease.forward->residency.prepare_residency_sliding(
      device, roles, memory, control);
  record_status(lease.observation,
                &ProductRouteObservation::prepare_residency_sliding_status,
                result);
  record_flag(lease.observation,
              lease.observation->prepare_residency_sliding_called);
  return result;
}

Status ObserveSubmitResidencySliding(
    const DeviceState &device,
    const rund::node::accel::detail::PreparedResidencySlidingRequest &request,
    rund::node::accel::detail::PreparedResidencySlidingControl
        &control) noexcept {
  CallbackLease lease;
  if (!begin_callback(const_cast<DeviceState *>(&device), lease) ||
      lease.forward->residency.submit_residency_sliding == nullptr) {
    return Status::fail(rund::compute::Reason::PipelineBusy);
  }
  const Status result = lease.forward->residency.submit_residency_sliding(
      device, request, control);
  record_status(lease.observation,
                &ProductRouteObservation::submit_residency_sliding_status,
                result);
  record_nonowning_accept(lease.observation,
                          lease.observation->submit_residency_sliding_called,
                          lease.observation->submit_residency_sliding_accepted,
                          static_cast<bool>(result));
  return result;
}

} // namespace rund_node_test_virtual::product::route_detail
