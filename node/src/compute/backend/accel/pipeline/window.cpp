#include "../../../device/state.hpp"
#include "../../../../accel/kernel/prepared/interface/api.hpp"
#include "../pipeline.hpp"

#include "../../../../accel/context/internal/support.hpp"
#include "../../../../accel/context/local.hpp"
#include "../../../status.hpp"

#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <memory>
#include <utility>

namespace rund::compute::detail::accel_backend {

Status submit_residency_window(
    const DeviceState &device,
    const node::accel::detail::PreparedResidencyWindowRequest &request,
    node::accel::detail::PreparedResidencyWindowControl &control) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck submitted =
      node::accel::detail::SubmitPreparedKernelPipelineWindow(accel->context,
                                                              request, control);
  return submitted.ok ? Status::success()
                      : Status::fail(project_reason(
                            submitted.reason, Reason::BackendUnsupported));
}

Status submit_residency_stream_window(
    const DeviceState &device,
    const node::accel::detail::PreparedResidencyWindowRequest &request,
    node::accel::detail::PreparedResidencyWindowControl &window,
    node::accel::detail::PreparedResidencyStreamControl &stream) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck submitted =
      node::accel::detail::SubmitPreparedKernelPipelineStreamWindow(
          accel->context, request, window, stream);
  return submitted.ok ? Status::success()
                      : Status::fail(project_reason(
                            submitted.reason, Reason::BackendUnsupported));
}

Status claim_residency_stream(
    const DeviceState &device,
    const std::span<const node::accel::detail::PreparedKernelPipeline>
        pipelines,
    const std::uint64_t plan_identity, const std::uint64_t token,
    const std::uint64_t generation,
    node::accel::detail::PreparedResidencyStreamControl &control) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck claimed =
      node::accel::detail::ClaimPreparedKernelPipelineStream(
          accel->context, pipelines, plan_identity, token, generation, control);
  return claimed.ok ? Status::success()
                    : Status::fail(project_reason(claimed.reason,
                                                  Reason::BackendUnsupported));
}

Status release_residency_stream(
    const DeviceState &device,
    node::accel::detail::PreparedResidencyStreamControl &control,
    const std::uint64_t plan_identity, const std::uint64_t token,
    const std::uint64_t generation, const bool quarantine) noexcept {
  if (accel_device(device) == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck released =
      node::accel::detail::ReleasePreparedKernelPipelineStream(
          control, plan_identity, token, generation, quarantine);
  return released.ok ? Status::success()
                     : Status::fail(project_reason(released.reason,
                                                   Reason::BackendFailed));
}

Status quarantine_residency_stream(
    const DeviceState &device,
    node::accel::detail::PreparedResidencyStreamControl &control,
    const std::uint64_t plan_identity, const std::uint64_t token,
    const std::uint64_t generation) noexcept {
  if (accel_device(device) == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck detached =
      node::accel::detail::QuarantinePreparedKernelPipelineStream(
          control, plan_identity, token, generation);
  return detached.ok ? Status::success()
                     : Status::fail(project_reason(detached.reason,
                                                   Reason::BackendFailed));
}

Status residency_window_capability(
    const DeviceState &device,
    const std::span<const node::accel::detail::PreparedKernelPipeline>
        pipelines,
    bool &ready) noexcept {
  ready = false;
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck capability =
      node::accel::detail::PreparedKernelPipelineWindowReady(accel->context,
                                                             pipelines, ready);
  return capability.ok ? Status::success()
                       : Status::fail(project_reason(
                             capability.reason, Reason::BackendUnsupported));
}

Status signal_residency_window(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    const node::accel::detail::BackendResidencyWindowSignal &signal) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck signalled =
      node::accel::detail::SignalPreparedKernelPipelineWindow(accel->context,
                                                              pipeline, signal);
  return signalled.ok ? Status::success()
                      : Status::fail(project_reason(signalled.reason,
                                                    Reason::BackendFailed));
}

Status abort_residency_window(
    const DeviceState &device,
    node::accel::detail::PreparedResidencyWindowControl &control,
    const node::accel::detail::BackendResidencyWindowAbort &abort) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck drained =
      node::accel::detail::AbortPreparedKernelPipelineWindow(accel->context,
                                                             control, abort);
  return drained.ok ? Status::success()
                    : Status::fail(project_reason(drained.reason,
                                                  Reason::BackendFailed));
}

} // namespace rund::compute::detail::accel_backend
