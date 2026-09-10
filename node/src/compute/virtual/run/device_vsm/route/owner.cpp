#include "../route.hpp"

#include "../../backing_set.hpp"
#include "../internal.hpp"
#include "../operations.hpp"

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

[[nodiscard]] bool scope_ok(const VirtualRunResources &resources,
                            const bool pooled) noexcept {
  return resources.view_commit == nullptr &&
         (pooled ? resources.pool.owns_lock() : !resources.pool.owns_lock());
}

[[nodiscard]] Status
discard_owner(VirtualPipelineState &state, VirtualRunResources &resources,
              VirtualExecutionDeviceVsmPrepared &prepared,
              const std::shared_ptr<DeviceVsmProductOwner> &owner,
              const bool pooled) noexcept {
  if (!scope_ok(resources, pooled)) {
    quarantine_owner(owner);
    return Status::fail(Reason::DeviceLost);
  }
  if (owner == nullptr) {
    prepared = {};
    return Status::success();
  }
  const auto cached = std::static_pointer_cast<DeviceVsmProductOwner>(
      state.device_vsm_product_cache);
  if (cached != owner || owner->registration == nullptr ||
      owner->release_registration == nullptr || owner_lost(owner)) {
    quarantine_owner(owner);
    return Status::fail(Reason::DeviceLost);
  }
  const residency::RegistrationResult released =
      owner->release_registration(owner->registration.get());
  if (released != residency::RegistrationResult::Done) {
    quarantine_owner(owner);
    return Status::fail(Reason::DeviceLost);
  }
  const auto still_cached = std::static_pointer_cast<DeviceVsmProductOwner>(
      state.device_vsm_product_cache);
  if (still_cached != owner || owner_lost(owner)) {
    quarantine_owner(owner);
    return Status::fail(Reason::DeviceLost);
  }
  state.device_vsm_product_cache.reset();
  prepared = {};
  owner->registration.reset();
  return Status::success();
}

[[nodiscard]] bool
rearm_owner(const std::shared_ptr<DeviceVsmProductOwner> &owner, Status &status,
            const char *&reason, bool &mutated) noexcept {
  mutated = false;
  const node::accel::detail::DeviceVsmRearmResult rearmed =
      owner->preparation.rearm(owner->preparation.lowering, owner->proof);
  mutated = rearmed.mutated;
  if (!rearmed.check.ok || !rearmed.mutated) {
    reason = rearmed.check.reason;
    if (rearmed.check.ok) {
      status = Status::fail(Reason::BackendFailed);
      reason = "compute_device_rearm_incomplete";
      return false;
    }
    status = status_from(rearmed.check);
    if (rearmed.mutated) {
      status = Status::fail(Reason::DeviceLost);
      reason = "compute_device_lost";
    }
    return false;
  }
  owner->submitted = false;
  ++owner->warm_rearm_count;
  return true;
}

} // namespace

Status rearm_prepared(VirtualExecutionDeviceVsmPrepared &prepared) noexcept {
  if (prepared.origin == VirtualDeviceVsmPreparedOrigin::ColdNew) {
    return Status::success();
  }
  const auto owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(prepared.owner);
  if (owner == nullptr || owner_lost(owner) || !owner->submitted) {
    prepared.reason = "compute_device_lost";
    return Status::fail(Reason::DeviceLost);
  }
  Status status = Status::fail(Reason::BackendFailed);
  const char *reason = "compute_backend_unsupported";
  bool mutated = false;
  if (!rearm_owner(owner, status, reason, mutated)) {
    prepared.rearm_mutated = mutated;
    prepared.reason = reason;
    return status;
  }
  prepared.rearm_mutated = mutated;
  prepared.reason = "ok";
  return Status::success();
}

Status discard_prepared(VirtualPipelineState &state,
                        VirtualRunResources &resources,
                        VirtualExecutionDeviceVsmPrepared &prepared,
                        const bool pooled) noexcept {
  const auto owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(prepared.owner);
  if (!scope_ok(resources, pooled)) {
    quarantine_owner(owner);
    return Status::fail(Reason::DeviceLost);
  }
  if (owner == nullptr) {
    prepared = {};
    return Status::success();
  }
  if (prepared.origin == VirtualDeviceVsmPreparedOrigin::WarmCached &&
      !prepared.rearm_mutated && owner->submitted) {
    const auto cached = std::static_pointer_cast<DeviceVsmProductOwner>(
        state.device_vsm_product_cache);
    if (cached != owner || owner_lost(owner)) {
      quarantine_owner(owner);
      return Status::fail(Reason::DeviceLost);
    }
    prepared = {};
    return Status::success();
  }
  return discard_owner(state, resources, prepared, owner, pooled);
}

} // namespace rund::compute::detail::device_vsm_product_detail
