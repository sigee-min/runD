#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::compute::detail::residency {

namespace {

[[nodiscard]] ResidentRecurrenceBinding
from_credential(const registration_detail::State::Binding &binding) noexcept {
  return ResidentRecurrenceBinding{
      .view = ResidentRecurrenceView{.resource = binding.resource,
                                     .bytes = binding.bytes,
                                     .offset_bytes = binding.offset_bytes,
                                     .element_bytes = binding.element_bytes,
                                     .stride_bytes = binding.stride_bytes,
                                     .count = binding.count,
                                     .usage = binding.usage},
      .region = FrameRegion{.tier = static_cast<FrameTier>(binding.tier),
                            .role = static_cast<FrameRole>(binding.role),
                            .first = binding.region_first,
                            .count = binding.region_count},
      .registration = binding.registration};
}

} // namespace

DirectRecurrenceLease DirectRecurrenceOwner::begin_direct_recurrence(
    const DirectRecurrenceRequest &request) noexcept {
  DirectRecurrenceLease lease{};
  if (request.proof_owner == nullptr || request.registration_state == nullptr ||
      (request.proof_hi == 0u && request.proof_lo == 0u) ||
      request.iterations < 2u ||
      request.iterations > std::numeric_limits<std::uint32_t>::max() ||
      request.registration_state->binding_count() == 0u ||
      request.registration_state->binding_count() >
          registry_model::ExecutionBindingCapacity) {
    return lease;
  }

  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_state_locked() !=
          registry_model::ViewCommitState::Idle ||
      !request.registration_state->owner_matches(request.proof_owner)) {
    return lease;
  }
  const std::shared_ptr<const void> sealed_owner =
      request.registration_state->owner();
  if (sealed_owner == nullptr || request.registration_state->phase() !=
                                     registration_detail::Lifecycle::Active) {
    return lease;
  }
  if (authority_.execution_state_.slot.token != 0u ||
      authority_.cycle_state_.cycle.token != 0u ||
      direct_recurrence_detail::active(authority_.cycle_state_.writeback) ||
      std::any_of(authority_.cycle_state_.epochs.begin(),
                  authority_.cycle_state_.epochs.end(),
                  direct_recurrence_detail::active)) {
    return lease;
  }

  registry_model::ExecutionSlot candidate{};
  candidate.direct_proof_owner = sealed_owner;
  candidate.registration_state = request.registration_state;
  candidate.registration_nonce = request.registration_state->nonce();
  candidate.direct_binding_count = request.registration_state->binding_count();
  for (std::size_t index = 0u; index < candidate.direct_binding_count;
       ++index) {
    candidate.direct_bindings[index] =
        from_credential(request.registration_state->binding(index));
  }
  const auto reserve = [this, &candidate](
                           const ResidentRecurrenceBinding &binding) noexcept {
    const std::size_t index = binding.region.first;
    constexpr std::uint32_t read = rund::kernel::kResidentUsageRead;
    constexpr std::uint32_t write = rund::kernel::kResidentUsageWrite;
    const std::uint32_t usage = binding.view.usage;
    const bool reads = (usage & read) != 0u;
    const bool writes = (usage & write) != 0u;
    if ((!reads && !writes) || (usage & ~(read | write)) != 0u ||
        index >= authority_.frames_.size() ||
        !direct_recurrence_detail::exact_view(authority_.frames_[index],
                                              binding) ||
        authority_.frames_[index].direct_registration !=
            candidate.registration_state ||
        (reads && !writes && binding.region.role != FrameRole::Input &&
         binding.region.role != FrameRole::Intermediate) ||
        (writes && !reads && binding.region.role != FrameRole::Output &&
         binding.region.role != FrameRole::Intermediate) ||
        (reads && writes && binding.region.role != FrameRole::Intermediate) ||
        (reads && authority_.frames_[index].state != FrameState::Resident) ||
        (!reads && writes &&
         authority_.frames_[index].state != FrameState::Resident &&
         authority_.frames_[index].state != FrameState::Empty) ||
        std::find(candidate.frames.begin(),
                  candidate.frames.begin() + candidate.frame_count,
                  static_cast<std::uint32_t>(index)) !=
            candidate.frames.begin() + candidate.frame_count) {
      return false;
    }
    candidate.frames[candidate.frame_count] = static_cast<std::uint32_t>(index);
    candidate.undo_frames[candidate.undo_count] =
        static_cast<std::uint32_t>(index);
    candidate.undo[candidate.undo_count] = authority_.frames_[index];
    candidate.service_bindings[0u][candidate.frame_count] = CacheBinding{
        .frame = static_cast<std::uint32_t>(index),
        .access = writes ? Access::Write : Access::Read,
    };
    ++candidate.frame_count;
    ++candidate.undo_count;
    return true;
  };

  for (std::size_t index = 0u; index < candidate.direct_binding_count;
       ++index) {
    if (!reserve(candidate.direct_bindings[index])) {
      return lease;
    }
  }
  if (!request.registration_state->transition(
          registration_detail::Lifecycle::Active,
          registration_detail::Lifecycle::Admitted)) {
    return lease;
  }
  candidate.service_binding_count[0u] = candidate.frame_count;

  candidate.token =
      direct_recurrence_detail::mint_local(authority_.credentials_.next_token);
  candidate.generation =
      direct_recurrence_detail::mint_local(authority_.credentials_.next_token);
  candidate.plan = request.proof_lo;
  candidate.epochs = request.iterations;
  candidate.native_inflight = request.proof_hi;
  candidate.next_sequence = direct_recurrence_detail::next_owner();
  candidate.direct_recurrence_admitted = true;
  for (std::size_t index = 0u; index < candidate.frame_count; ++index) {
    authority_.frames_[candidate.frames[index]].state = FrameState::Pinned;
  }
  authority_.execution_state_.slot = std::move(candidate);

  lease.proof_owner_ = sealed_owner;
  lease.proof_hi_ = request.proof_hi;
  lease.proof_lo_ = request.proof_lo;
  lease.token_ = authority_.execution_state_.slot.token;
  lease.generation_ = authority_.execution_state_.slot.generation;
  lease.owner_ = authority_.execution_state_.slot.next_sequence;
  lease.iterations_ = request.iterations;
  lease.registration_state_ = request.registration_state;
  lease.registration_nonce_ = request.registration_state->nonce();
  return lease;
}

} // namespace rund::compute::detail::residency
