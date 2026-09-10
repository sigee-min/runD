#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {
namespace {

[[nodiscard]] Status build_role_set(const SlidingProductOwner &owner,
                                    const PipelineExecutionSnapshot &snapshot,
                                    SlidingProductAttempt &attempt) noexcept {
  const bool transactional = owner.pipelines[0u]->transactional;
  if (owner.pipelines[1u]->transactional != transactional) {
    return Status::fail(Reason::BackendUnsupported);
  }
  // W=4 bank/parity recurrence remains outside the product evidence surface.
  if (transactional) {
    return Status::fail(Reason::BackendUnsupported);
  }
  attempt = {};
  attempt.snapshot = snapshot;
  attempt.role_count = 2u;
  attempt.generation_step = 1u;
  if (!control_range(attempt.snapshot, attempt.role_count,
                     owner.plan.epoch_count())) {
    return Status::fail(Reason::PipelineCapacity);
  }
  for (std::size_t slot = 0u; slot < attempt.role_count; ++slot) {
    const std::size_t bank = slot % residency::execution::BankCapacity;
    const std::uint8_t parity = static_cast<std::uint8_t>(
        attempt.snapshot.parity[bank] ^
        static_cast<std::uint8_t>(slot / residency::execution::BankCapacity));
    const node::accel::detail::PreparedKernelPipeline *const native =
        prepared_pipeline_for(*owner.pipelines[bank], parity);
    if (native == nullptr || !native->ok) {
      return Status::fail(Reason::PipelineInvalid);
    }
    attempt.roles[slot] = node::accel::detail::PreparedResidencySlidingRole{
        .pipeline = *native,
        .first_control_generation = static_cast<std::uint32_t>(
            attempt.snapshot.generation[bank] +
            slot / residency::execution::BankCapacity + 1u),
        .control_generation_stride = attempt.generation_step,
        .slot = static_cast<std::uint8_t>(slot),
    };
    node::accel::detail::PreparedResidencyPersistentSlidingRole persistent{
        .pipeline = *native,
        .local_count = owner.plan.frame_capacity(),
        .first_control_generation =
            attempt.roles[slot].first_control_generation,
        .control_generation_stride = attempt.generation_step,
        .first_descriptor_generation = 1u,
        .descriptor_generation_stride = 1u,
        .slot = static_cast<std::uint8_t>(slot),
    };
    if (persistent.local_count == 0u ||
        persistent.local_count > persistent.locals.size()) {
      return Status::fail(Reason::PipelineCapacity);
    }
    for (std::size_t local = 0u; local < persistent.local_count; ++local) {
      persistent.locals[local] = static_cast<std::uint32_t>(local);
    }
    attempt.persistent_roles[slot] = std::move(persistent);
  }
  return Status::success();
}

} // namespace

Status
stage_preparation_roles(SlidingProductOwner &owner,
                        const PipelineExecutionSnapshot &snapshot) noexcept {
  SlidingProductAttempt attempt{};
  const Status status = build_role_set(owner, snapshot, attempt);
  if (!status) {
    return status;
  }
  owner.pending = std::move(attempt);
  owner.pending_valid = true;
  return Status::success();
}

void commit_staged_roles(SlidingProductOwner &owner) noexcept {
  if (!owner.pending_valid) {
    return;
  }
  owner.snapshot = owner.pending.snapshot;
  owner.roles = owner.pending.roles;
  owner.persistent_roles = owner.pending.persistent_roles;
  owner.role_count = owner.pending.role_count;
  owner.generation_step = owner.pending.generation_step;
  owner.pending = {};
  owner.pending_valid = false;
}

void discard_staged_roles(SlidingProductOwner &owner) noexcept {
  owner.pending = {};
  owner.pending_valid = false;
}

Status build_preparation_roles(SlidingProductOwner &owner) noexcept {
  const Status status = stage_preparation_roles(owner, owner.snapshot);
  if (status) {
    commit_staged_roles(owner);
  }
  return status;
}

} // namespace rund::compute::detail::sliding_product_detail
