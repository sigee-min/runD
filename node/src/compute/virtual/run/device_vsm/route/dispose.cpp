#include "../route.hpp"

#include "../../backing_set.hpp"
#include "../internal.hpp"
#include "../operations.hpp"

namespace rund::compute::detail {

Status dispose_virtual_device_vsm_route(
    VirtualPipelineState &state, VirtualRunResources &resources,
    VirtualDeviceVsmPostStage &post, const VirtualDeviceVsmScope scope,
    const Status failure, bool &poison,
    VirtualRunWriteCertainty &certainty) noexcept {
  using namespace device_vsm_product_detail;
  const auto owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(post.prepared.owner);
  const bool pooled = scope == VirtualDeviceVsmScope::Pooled;
  bool unknown =
      failure.reason() == Reason::DeviceLost || post.prepared.rearm_mutated;
  bool receipt_unknown = false;
  const bool lock_ok =
      pooled ? resources.pool.owns_lock() : !resources.pool.owns_lock();
  if (!lock_ok) {
    poison = true;
    certainty = VirtualRunWriteCertainty::UnknownMayWrite;
    if (resources.view_commit != nullptr) {
      static_cast<void>(quarantine_pool_stage(resources));
    }
    quarantine_owner(owner);
    return Status::fail(Reason::DeviceLost);
  }
  if (resources.view_commit != nullptr) {
    const Status quarantined = quarantine_pool_stage(resources);
    unknown = true;
    receipt_unknown = true;
    poison = true;
    if (resources.view_commit != nullptr) {
      certainty = VirtualRunWriteCertainty::UnknownMayWrite;
      return quarantined;
    }
  }
  if (failure.reason() == Reason::DeviceLost) {
    quarantine_owner(owner);
  }
  const Status discarded =
      discard_prepared(state, resources, post.prepared, pooled);
  if (!discarded) {
    poison = true;
    certainty = VirtualRunWriteCertainty::UnknownMayWrite;
    return discarded;
  }
  if (unknown) {
    quarantine_owner(owner);
    poison = true;
    certainty = VirtualRunWriteCertainty::UnknownMayWrite;
  } else {
    certainty = VirtualRunWriteCertainty::KnownNoWrite;
  }
  return receipt_unknown ? Status::fail(Reason::DeviceLost) : failure;
}

} // namespace rund::compute::detail
