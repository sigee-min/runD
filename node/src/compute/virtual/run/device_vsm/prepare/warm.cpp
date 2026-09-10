#include "warm/local.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

std::shared_ptr<DeviceVsmProductOwner>
prepare_warm_owner(VirtualPipelineState &state, const VirtualRunProjection &run,
                   const node::accel::detail::DeviceVsmIdentity &identity,
                   const VirtualDeviceVsmRouteProof &proof, Status &status,
                   const char *&reason) noexcept {
  const AccelDeviceState *const native =
      state.pipeline == nullptr || state.pipeline->device == nullptr
          ? nullptr
          : accel_device(*state.pipeline->device);
  const auto owner = std::static_pointer_cast<DeviceVsmProductOwner>(
      state.device_vsm_product_cache);
  if (!proof.valid()) {
    status = Status::fail(Reason::BackendUnsupported);
    reason = "compute_backend_unsupported";
    return {};
  }
  if (owner != nullptr && owner_lost(owner)) {
    status = Status::fail(Reason::DeviceLost);
    reason = "compute_device_lost";
    return {};
  }
  if (!warm_detail::owner_ready(owner.get()) ||
      !warm_detail::owner_shape_matches(state, run, identity, proof, *owner)) {
    status = Status::fail(Reason::BackendUnsupported);
    reason = "compute_backend_unsupported";
    return {};
  }
  if (!owner->submitted) {
    status = Status::fail(Reason::BackendUnsupported);
    reason = "compute_backend_unsupported";
    return {};
  }
  if (!warm_detail::backend_bindings_match(state, run, native, *owner)) {
    status = Status::fail(Reason::BackendUnsupported);
    reason = "device_vsm_graph_resident_external_binding_invalid";
    return {};
  }
  if (!warm_detail::pipelines_match(state, run, *owner)) {
    status = Status::fail(Reason::BackendUnsupported);
    reason = "compute_backend_unsupported";
    return {};
  }
  if (owner->proof->topology ==
          node::accel::detail::DeviceVsmTopology::GraphPointwise &&
      !warm_detail::validate_pointwise(state, run, *owner)) {
    status = Status::fail(Reason::BackendUnsupported);
    reason = "compute_device_vsm_graph_pointwise_page_map_changed";
    return {};
  }
  if (owner->proof->topology ==
          node::accel::detail::DeviceVsmTopology::GraphResident &&
      !warm_detail::validate_resident(state, run, *owner, reason)) {
    status = Status::fail(Reason::BackendUnsupported);
    return {};
  }
  status = Status::success();
  reason = "ok";
  return owner;
}

} // namespace rund::compute::detail::device_vsm_product_detail
