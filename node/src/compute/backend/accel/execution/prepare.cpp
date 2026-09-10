#include "../../../../accel/kernel/prepared/interface/api.hpp"
#include "../local.hpp"

#include "../../../status.hpp"
#include "../../../device/residency/execution/owner.hpp"

namespace rund::compute::detail::accel_backend {

Status prepare_residency_execution(
    const DeviceState &,
    const std::span<const node::accel::detail::PreparedKernelPipeline *const>
        pipelines,
    residency::execution::Owner &owner) noexcept {
  owner = {};
  if (pipelines.size() != 1u || pipelines.front() == nullptr ||
      !pipelines.front()->ok) {
    return Status::fail(Reason::BackendUnsupported);
  }

  bool ready = false;
  const rund::AccelCheck capability =
      node::accel::detail::PreparedKernelPipelineResidencyReady(
          *pipelines.front(), ready);
  if (!capability.ok || !ready) {
    return Status::fail(Reason::BackendUnsupported);
  }

  // The prepared Pipeline already owns and accounts its retained native
  // command resources and async submission slot. Binding this shared owner is
  // a refcount operation, not a run allocation.
  owner = residency::execution::Owner{
      .native = pipelines.front()->owner,
  };
  return Status::success();
}

} // namespace rund::compute::detail::accel_backend
