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

Status prepare_residency_sliding(
    const DeviceState &device,
    const std::span<const node::accel::detail::PreparedResidencySlidingRole>
        roles,
    const node::accel::detail::ResidencySlidingMemory memory,
    node::accel::detail::PreparedResidencySlidingControl &control) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck prepared =
      node::accel::detail::PrepareKernelPipelineSliding(accel->context, roles,
                                                        memory, control);
  return prepared.ok ? Status::success()
                     : Status::fail(project_reason(prepared.reason,
                                                   Reason::BackendUnsupported));
}

Status submit_residency_sliding(
    const DeviceState &device,
    const node::accel::detail::PreparedResidencySlidingRequest &request,
    node::accel::detail::PreparedResidencySlidingControl &control) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck submitted =
      node::accel::detail::SubmitPreparedKernelPipelineSliding(
          accel->context, request, control);
  return submitted.ok ? Status::success()
                      : Status::fail(project_reason(
                            submitted.reason, Reason::BackendUnsupported));
}

Status wake_residency_sliding(
    const DeviceState &device,
    node::accel::detail::PreparedResidencySlidingControl &control) noexcept {
  if (accel_device(device) == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const rund::AccelCheck woken =
      node::accel::detail::WakePreparedKernelPipelineSliding(control);
  return woken.ok ? Status::success()
                  : Status::fail(
                        project_reason(woken.reason, Reason::BackendFailed));
}

} // namespace rund::compute::detail::accel_backend
