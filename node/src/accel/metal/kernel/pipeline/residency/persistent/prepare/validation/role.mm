#include "../../../../../../../kernel/prepared/interface/api.hpp"
#include "../../../../../../../kernel/prepared/model.hpp"
#include "../../internal.hpp"

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

NativeRole
native_role(const PreparedResidencyPersistentSlidingRole &source) noexcept {
  NativeRole role{
      .locals = source.locals,
      .local_count = source.local_count,
      .first_control_generation = source.first_control_generation,
      .control_generation_stride = source.control_generation_stride,
      .first_descriptor_generation = source.first_descriptor_generation,
      .descriptor_generation_stride = source.descriptor_generation_stride,
      .slot = source.slot,
      .pipeline_ok = source.pipeline.ok,
  };
  if (!source.pipeline.ok || source.pipeline.owner == nullptr) {
    return role;
  }
  const auto *const state =
      static_cast<const prepared::PipelineState *>(source.pipeline.owner.get());
  if (state != nullptr && state->backend != nullptr) {
    role.sequence = static_cast<const MetalSequence *>(state->backend.get());
  }
  return role;
}

std::uint64_t first_invalid_structure(const std::span<const NativeRole> roles,
                                      const std::size_t width,
                                      const std::uint64_t coordinate_count,
                                      const ResidencySlidingMemory memory,
                                      const PersistentResidencySlidingMode mode,
                                      MetalAdapter *&adapter) noexcept {
  adapter = nullptr;
  if (coordinate_count == 0u)
    return request_issue_key(RequestIssue::CoordinateCount);
  if (memory != ResidencySlidingMemory::HostCoherent)
    return request_issue_key(RequestIssue::Memory);
  if (mode != PersistentResidencySlidingMode::OneSubmit &&
      mode != PersistentResidencySlidingMode::BackendChunked)
    return request_issue_key(RequestIssue::Mode);
  if (width == 0u)
    return request_issue_key(RequestIssue::WidthZero);
  if (width > PersistentResidencySlidingCapacity || width > roles.size())
    return request_issue_key(RequestIssue::WidthCapacity);
  for (std::size_t slot = 0u; slot < width; ++slot) {
    const NativeRole &role = roles[slot];
    const MetalSequence *const sequence = role.sequence;
    const bool spatial_window_route =
        sequence != nullptr && sequence->persistent_spatial_window_selectable;
    if (role.slot != slot)
      return request_issue_key(RequestIssue::RoleSlot, slot);
    if (role.local_count == 0u)
      return request_issue_key(RequestIssue::RoleLocalZero, slot);
    if (role.local_count > ResidencyWindowLocalCapacity)
      return request_issue_key(RequestIssue::RoleLocalCapacity, slot);
    if (role.first_control_generation == 0u)
      return request_issue_key(RequestIssue::RoleControlGeneration, slot);
    if (role.control_generation_stride == 0u)
      return request_issue_key(RequestIssue::RoleControlStride, slot);
    if (role.first_descriptor_generation == 0u)
      return request_issue_key(RequestIssue::RoleDescriptorGeneration, slot);
    if (role.descriptor_generation_stride == 0u)
      return request_issue_key(RequestIssue::RoleDescriptorStride, slot);
    if (!role.pipeline_ok || !ValidMetalSequence(sequence))
      return request_issue_key(RequestIssue::RoleSequence, slot);
    if (sequence->adapter == nullptr)
      return request_issue_key(RequestIssue::RoleAdapter, slot);
    for (std::size_t prior = 0u; prior < slot; ++prior) {
      if (roles[prior].sequence == role.sequence)
        return request_issue_key(RequestIssue::RoleSequence, slot);
    }
    if (sequence->direct_aggregate)
      return request_issue_key(RequestIssue::RoleDirectAggregate, slot);
    if (sequence->state_count != 0u)
      return request_issue_key(RequestIssue::RoleStateCount, slot);
    if (sequence->states != nil)
      return request_issue_key(RequestIssue::RoleStates, slot);
    if (sequence->recurrence != nullptr)
      return request_issue_key(RequestIssue::RoleRecurrence, slot);
    if (sequence->spatial_window.window_present && !spatial_window_route)
      return request_issue_key(RequestIssue::RoleSpatialRoute, slot);
    if (spatial_window_route && !sequence->spatial_window_proof_valid())
      return request_issue_key(RequestIssue::RoleSpatialProof, slot);
    if (spatial_window_route && sequence->state_count != 0u)
      return request_issue_key(RequestIssue::RoleSpatialStateCount, slot);
    if (spatial_window_route && sequence->states != nil)
      return request_issue_key(RequestIssue::RoleSpatialStates, slot);
    if (spatial_window_route && sequence->recurrence != nullptr)
      return request_issue_key(RequestIssue::RoleSpatialRecurrence, slot);
    if (!sequence->residency_selectable)
      return request_issue_key(RequestIssue::RoleSelectable, slot);
    if (sequence->guard_zero == nil)
      return request_issue_key(RequestIssue::RoleGuard, slot);
    if ([sequence->guard_zero contents] == nullptr)
      return request_issue_key(RequestIssue::RoleGuardContents, slot);
    if (sequence->control == nil)
      return request_issue_key(RequestIssue::RoleControl, slot);
    if ([sequence->control contents] == nullptr)
      return request_issue_key(RequestIssue::RoleControlContents, slot);
    if (!sequence->residency_submission.ready())
      return request_issue_key(RequestIssue::RoleSubmissionReady, slot);
    if (!sequence->residency_sliding.ready_for_submit)
      return request_issue_key(RequestIssue::RoleSlidingReady, slot);
    if (sequence->residency_sliding.descriptor == nil)
      return request_issue_key(RequestIssue::RoleDescriptor, slot);
    if (sequence->residency_sliding.command == nil)
      return request_issue_key(RequestIssue::RoleCommand, slot);
    if (sequence->residency_sliding.pipeline == nil)
      return request_issue_key(RequestIssue::RolePipeline, slot);
    if (adapter != nullptr && adapter != sequence->adapter)
      return request_issue_key(RequestIssue::RoleAdapterMismatch, slot);
    for (std::size_t local = 0u; local < role.local_count; ++local) {
      if (role.locals[local] >= sequence->residency_steps.size())
        return request_issue_key(RequestIssue::LocalStepBounds, slot, local);
      if (spatial_window_route &&
          role.locals[local] >= sequence->spatial_window.local_count)
        return request_issue_key(RequestIssue::LocalSpatialBounds, slot, local);
      if (spatial_window_route &&
          !sequence->spatial_window.local_matches(role.locals[local]))
        return request_issue_key(RequestIssue::LocalSpatialMatch, slot, local);
      for (std::size_t prior = 0u; prior < local; ++prior) {
        if (role.locals[prior] == role.locals[local])
          return request_issue_key(RequestIssue::LocalDuplicate, slot, local);
      }
    }
    adapter = sequence->adapter;
  }
  if (!persistent_sliding_range_valid(roles, width, coordinate_count))
    return request_issue_key(RequestIssue::GenerationRange);
  if (adapter == nullptr)
    return request_issue_key(RequestIssue::TailAdapter);
  if (adapter->queue == nullptr)
    return request_issue_key(RequestIssue::TailQueue);
  if (adapter->residency_quarantined.load(std::memory_order_acquire))
    return request_issue_key(RequestIssue::AdapterQuarantined);
  return request_issue_key(RequestIssue::Valid);
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
