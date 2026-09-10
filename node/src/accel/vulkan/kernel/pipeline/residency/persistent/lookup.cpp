#include "internal.hpp"

namespace rund::node::accel::detail::vulkan_persistent_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

std::shared_ptr<Owner> owner_of(const std::shared_ptr<void> &opaque) noexcept {
  auto owner = std::static_pointer_cast<Owner>(opaque);
  return owner != nullptr && owner->magic == OwnerMagic ? std::move(owner)
                                                        : nullptr;
}

bool same_prepared_request(
    const Owner &owner,
    const PersistentResidencySlidingRequest &request) noexcept {
  const PersistentResidencySlidingRequest &prepared = owner.prepared;
  if (request.plan_identity != prepared.plan_identity ||
      request.token != prepared.token ||
      request.generation != prepared.generation ||
      request.owner_nonce != prepared.owner_nonce ||
      request.cell_id != prepared.cell_id ||
      request.cell_domain != prepared.cell_domain ||
      request.coordinate_count != prepared.coordinate_count ||
      request.tail_local_count != prepared.tail_local_count ||
      request.admission != prepared.admission ||
      request.final != prepared.final || request.user != prepared.user ||
      request.memory != prepared.memory || request.width != prepared.width ||
      request.mode != prepared.mode) {
    return false;
  }
  for (std::size_t slot = 0u; slot < request.width; ++slot) {
    const PersistentResidencySlidingRole &left = request.roles[slot];
    const PersistentResidencySlidingRole &right = prepared.roles[slot];
    if (left.prepared != right.prepared || left.locals != right.locals ||
        left.local_count != right.local_count ||
        left.first_control_generation != right.first_control_generation ||
        left.control_generation_stride != right.control_generation_stride ||
        left.first_descriptor_generation != right.first_descriptor_generation ||
        left.descriptor_generation_stride !=
            right.descriptor_generation_stride ||
        left.slot != right.slot) {
      return false;
    }
  }
  return true;
}

VulkanResidencyPersistentRun *
active_run(PersistentResidencySlidingControl &control) noexcept {
  const std::shared_ptr<Owner> owner = owner_of(control.native);
  if (owner == nullptr || owner->run == nullptr) {
    return nullptr;
  }
  VulkanResidencyPersistentRun *const run = owner->run;
  std::lock_guard run_lock{run->gate};
  const PersistentResidencySlidingRequest &request = run->request;
  if (!run->active || run->control != &control ||
      request.plan_identity != control.plan_identity ||
      request.token != control.token ||
      request.generation != control.generation ||
      request.owner_nonce != control.owner_nonce) {
    return nullptr;
  }
  return run;
}

#endif

} // namespace rund::node::accel::detail::vulkan_persistent_detail
