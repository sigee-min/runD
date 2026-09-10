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

node::accel::detail::PreparedPipelineEvidence
run_pipeline(const DeviceState &device,
             const node::accel::detail::PreparedKernelPipeline &pipeline,
             const node::accel::detail::PipelineSubmitMode mode) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return node::accel::detail::PreparedPipelineEvidence{
        .check = {false, "accel_device_invalid"}};
  }
  return node::accel::detail::RunPreparedKernelPipeline(accel->context,
                                                        pipeline, mode);
}

rund::AccelCheck submit_residency_pipeline(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &prepared,
    const std::span<const std::uint32_t> locals, std::shared_ptr<void> lifetime,
    const node::accel::detail::PreparedPipelineCompletion completion,
    void *const user) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  return accel == nullptr
             ? rund::AccelCheck{false, "accel_device_invalid"}
             : node::accel::detail::SubmitPreparedKernelPipelineSelection(
                   accel->context, prepared, std::move(lifetime), completion,
                   user, node::accel::detail::KernelTiming::Submission,
                   node::accel::detail::PipelineSubmitMode::Residency, locals);
}

Status virtual_pipeline_capability(const DeviceState &device) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  const node::accel::detail::ContextAdmission admission =
      node::accel::detail::AdmitContextForSupport(accel->context);
  if (!admission.check.ok || admission.pick == nullptr ||
      admission.pick->ops == nullptr ||
      admission.pick->ops->virtual_pipeline_capability == nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  const rund::AccelCheck capability =
      admission.pick->ops->virtual_pipeline_capability(admission.pick->raw);
  return capability.ok ? Status::success()
                       : Status::fail(project_reason(
                             capability.reason, Reason::BackendUnsupported));
}

rund::AccelCheck submit_pipeline(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    std::shared_ptr<void> lifetime,
    const node::accel::detail::PreparedPipelineCompletion completion,
    void *const user, const node::accel::detail::KernelTiming timing,
    const node::accel::detail::PipelineSubmitMode mode) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  return accel == nullptr ? rund::AccelCheck{false, "accel_device_invalid"}
                          : node::accel::detail::SubmitPreparedKernelPipeline(
                                accel->context, pipeline, std::move(lifetime),
                                completion, user, timing, mode);
}

} // namespace rund::compute::detail::accel_backend
