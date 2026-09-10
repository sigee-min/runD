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

node::accel::detail::BackendResidencySchedulePreparation
prepare_residency_schedule(
    const DeviceState &device,
    const std::span<const node::accel::detail::PreparedResidencyScheduleRole>
        roles,
    const std::uint64_t epoch_count,
    const std::size_t tail_local_count) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  return accel == nullptr
             ? node::accel::detail::BackendResidencySchedulePreparation{}
             : node::accel::detail::PrepareKernelPipelineSchedule(
                   accel->context, roles, epoch_count, tail_local_count);
}

Status submit_residency_schedule(
    const DeviceState &device,
    const node::accel::detail::PreparedResidencyScheduleRequest &request,
    node::accel::detail::PreparedResidencyScheduleControl &schedule,
    node::accel::detail::PreparedResidencyStreamControl &stream) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck submitted =
      node::accel::detail::SubmitPreparedKernelPipelineSchedule(
          accel->context, request, schedule, stream);
  return submitted.ok ? Status::success()
                      : Status::fail(project_reason(
                            submitted.reason, Reason::BackendUnsupported));
}

Status signal_residency_schedule(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    const node::accel::detail::BackendResidencyWindowSignal &signal) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck signalled =
      node::accel::detail::SignalPreparedKernelPipelineSchedule(
          accel->context, pipeline, signal);
  return signalled.ok ? Status::success()
                      : Status::fail(project_reason(signalled.reason,
                                                    Reason::BackendFailed));
}

Status abort_residency_schedule(
    const DeviceState &device,
    node::accel::detail::PreparedResidencyScheduleControl &control,
    const node::accel::detail::BackendResidencyWindowAbort &abort) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck drained =
      node::accel::detail::AbortPreparedKernelPipelineSchedule(accel->context,
                                                               control, abort);
  return drained.ok ? Status::success()
                    : Status::fail(project_reason(drained.reason,
                                                  Reason::BackendFailed));
}

} // namespace rund::compute::detail::accel_backend
