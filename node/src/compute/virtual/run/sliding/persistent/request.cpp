#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

node::accel::detail::PersistentResidencySlidingRequest
make_persistent_request(SlidingProductRun &state,
                        const SlidingProductOwner &owner,
                        const residency::ExecutionLease lease) noexcept {
  const std::uint64_t remainder =
      owner.plan.page_count() % owner.plan.frame_capacity();
  const std::size_t tail = static_cast<std::size_t>(
      remainder == 0u ? owner.plan.frame_capacity() : remainder);
  const auto &roles = owner.pending_valid ? owner.pending.persistent_roles
                                          : owner.persistent_roles;
  const std::size_t role_count =
      owner.pending_valid ? owner.pending.role_count : owner.role_count;
  const auto committed =
      state.persistent_preparation.backend.cell != nullptr
          ? state.persistent_preparation.backend.cell->committed
          : node::accel::detail::PersistentResidencySlidingRequest{};
  const bool has_committed =
      state.persistent_preparation.backend.cell != nullptr &&
      committed.cell_id != 0u && committed.cell_domain != 0u;
  const bool committed_roles = has_committed && committed.width == role_count;
  node::accel::detail::PersistentResidencySlidingRequest request{
      .plan_identity = owner.plan.identity(),
      .token = lease.token,
      .generation = lease.generation,
      .owner_nonce = lease.owner_nonce,
      .cell_id = committed.cell_id,
      .cell_domain = committed.cell_domain,
      .coordinate_count = owner.plan.epoch_count(),
      .chunk_count = owner.mode ==
                             node::accel::detail::PersistentResidencySlidingMode::BackendChunked
                         ? 2u
                         : 0u,
      .tail_local_count = tail,
      .lowering = state.persistent_preparation.backend.lowering,
      .admission = owner.plan_owner,
      .final = final_persistent,
      .user = &state,
      .ticket = state.persistent_preparation.backend.ticket,
      .memory = node::accel::detail::ResidencySlidingMemory::HostCoherent,
      .width = static_cast<std::uint8_t>(role_count),
      .mode = owner.mode,
  };
  for (std::size_t slot = 0u; slot < role_count; ++slot) {
    const auto &role = roles[slot];
    request.roles[slot] = node::accel::detail::PersistentResidencySlidingRole{
        .prepared = has_committed
                        ? (committed_roles ? committed.roles[slot].prepared
                                           : std::shared_ptr<void>{})
                        : role.pipeline.owner,
        .locals = role.locals,
        .local_count = role.local_count,
        .first_control_generation = role.first_control_generation,
        .control_generation_stride = role.control_generation_stride,
        .first_descriptor_generation = role.first_descriptor_generation,
        .descriptor_generation_stride = role.descriptor_generation_stride,
        .slot = role.slot,
    };
  }
  return request;
}

} // namespace rund::compute::detail::sliding_product_detail
