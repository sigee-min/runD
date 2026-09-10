#include "internal.hpp"

#include "../backing_set.hpp"

#include "../../../../accel/context/local.hpp"

namespace rund::compute::detail {

Status prepare_accel_virtual_execution_device_vsm(
    VirtualPipelineState &state, const VirtualRunProjection &run,
    const VirtualDeviceVsmRouteProof proof,
    VirtualExecutionDeviceVsmPrepared &prepared) noexcept {
  using namespace device_vsm_product_detail;
  namespace accel = node::accel::detail;
  prepared = {};
  if (!proof.valid()) {
    return Status::fail(Reason::BackendUnsupported);
  }
  const AccelDeviceState *const native = accel_device(*state.pipeline->device);
  if (native == nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  if (!proof_matches(state, run, proof)) {
    prepared.reason = "compute_backend_unsupported";
    return Status::fail(Reason::BackendUnsupported);
  }
  const accel::DeviceVsmIdentity identity{.hi = proof.stamp_hi,
                                          .lo = proof.stamp_lo};
  const char *reason = "compute_backend_unsupported";
  Status status = Status::fail(Reason::BackendUnsupported);
  VirtualDeviceVsmPreparedOrigin origin =
      VirtualDeviceVsmPreparedOrigin::ColdNew;
  std::shared_ptr<DeviceVsmProductOwner> owner =
      prepare_warm_owner(state, run, identity, proof, status, reason);
  if (owner != nullptr) {
    origin = VirtualDeviceVsmPreparedOrigin::WarmCached;
  }
  if (owner == nullptr && status.reason() == Reason::BackendUnsupported) {
    const auto cached = std::static_pointer_cast<DeviceVsmProductOwner>(
        state.device_vsm_product_cache);
    if (cached != nullptr && cached->evidence != nullptr &&
        cached->evidence->quarantined) {
      prepared.reason = "compute_device_lost";
      return Status::fail(Reason::DeviceLost);
    }
    state.device_vsm_product_cache.reset();
    owner =
        prepare_cold_owner(state, run, *native, identity, proof, status, reason);
  }
  if (owner == nullptr) {
    prepared.reason = reason;
    return status;
  }
  state.device_vsm_product_cache = owner;
  prepared.owner = std::move(owner);
  prepared.reason = "ok";
  prepared.proof = proof;
  prepared.origin = origin;
  prepared.rearm_mutated = false;
  return Status::success();
}

} // namespace rund::compute::detail
