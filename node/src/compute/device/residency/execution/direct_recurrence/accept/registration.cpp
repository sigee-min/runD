#include "../internal.hpp"

#include "../../../registry/release_check.hpp"

#include <array>

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

[[nodiscard]] bool
exact(const Authority::Frame &frame,
      const registration_detail::State::Binding &binding,
      const std::shared_ptr<const registration_detail::State> &state) noexcept {
  return frame.direct_registration == state &&
         direct_recurrence_detail::exact_view(frame, from_credential(binding));
}

} // namespace

bool DirectRecurrenceOwner::bind_direct_registration(
    const std::shared_ptr<const registration_detail::State> &state) noexcept {
  if (state == nullptr || state->nonce() == 0u ||
      state->binding_count() == 0u ||
      state->binding_count() > registration_detail::State::BindingCapacity) {
    return false;
  }
  if (state->phase() != registration_detail::Lifecycle::Active) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_state_locked() !=
      registry_model::ViewCommitState::Idle) {
    return false;
  }
  std::array<std::uint32_t, registration_detail::State::BindingCapacity> rows{};
  for (std::size_t index = 0u; index < state->binding_count(); ++index) {
    const auto binding = from_credential(state->binding(index));
    const std::size_t frame = binding.region.first;
    if (frame >= authority_.frames_.size() || binding.region.count != 1u ||
        !direct_recurrence_detail::exact_view(authority_.frames_[frame],
                                              binding) ||
        authority_.frames_[frame].direct_registration != nullptr) {
      return false;
    }
    rows[index] = static_cast<std::uint32_t>(frame);
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (rows[prior] == rows[index]) {
        return false;
      }
    }
  }
  for (std::size_t index = 0u; index < state->binding_count(); ++index) {
    authority_.frames_[rows[index]].direct_registration = state;
  }
  return true;
}

RegistrationResult Authority::rollback_direct_registration(
    const std::span<const ResidentRecurrenceBinding> bindings) noexcept {
  if (bindings.empty() ||
      bindings.size() > registration_detail::State::BindingCapacity) {
    return RegistrationResult::Busy;
  }
  std::lock_guard lock{gate_};
  if (view_commit_state_locked() != registry_model::ViewCommitState::Idle) {
    return RegistrationResult::Busy;
  }
  std::array<std::uint32_t, registration_detail::State::BindingCapacity> rows{};
  std::array<FrameRegion, registration_detail::State::BindingCapacity>
      regions{};
  for (std::size_t index = 0u; index < bindings.size(); ++index) {
    const ResidentRecurrenceBinding &binding = bindings[index];
    const std::size_t frame = binding.region.first;
    if (frame >= frames_.size() || binding.region.count != 1u ||
        !direct_recurrence_detail::exact_view(frames_[frame], binding) ||
        frames_[frame].direct_registration != nullptr ||
        frames_[frame].alias_claims != 0u || !frames_[frame].dirty.empty() ||
        (frames_[frame].state != FrameState::Resident &&
         frames_[frame].state != FrameState::Empty)) {
      return RegistrationResult::Busy;
    }
    rows[index] = static_cast<std::uint32_t>(frame);
    regions[index] = binding.region;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (rows[prior] == rows[index]) {
        return RegistrationResult::Busy;
      }
    }
  }
  if (!release_detail::ReleaseCheck::allowed(
          *this, {regions.data(), bindings.size()})) {
    return RegistrationResult::Busy;
  }
  for (std::size_t index = 0u; index < bindings.size(); ++index) {
    frames_[rows[index]] = Frame{};
  }
  return RegistrationResult::Done;
}

RegistrationResult DirectRecurrenceOwner::retire_direct_registration(
    const std::shared_ptr<const registration_detail::State> &state) noexcept {
  if (state == nullptr || state->nonce() == 0u) {
    return RegistrationResult::Busy;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_state_locked() !=
      registry_model::ViewCommitState::Idle) {
    return RegistrationResult::Busy;
  }
  const registration_detail::Lifecycle phase = state->phase();
  if (phase == registration_detail::Lifecycle::Quarantined) {
    return RegistrationResult::Quarantined;
  }
  if (phase != registration_detail::Lifecycle::Active) {
    return RegistrationResult::Busy;
  }
  if (authority_.execution_state_.slot.token != 0u ||
      authority_.execution_state_.slot.direct_recurrence_admitted ||
      authority_.execution_state_.slot.registration_state != nullptr) {
    return RegistrationResult::Busy;
  }
  std::array<FrameRegion, registration_detail::State::BindingCapacity>
      regions{};
  for (std::size_t index = 0u; index < state->binding_count(); ++index) {
    const auto &binding = state->binding(index);
    const std::size_t frame = binding.region_first;
    if (frame >= authority_.frames_.size() ||
        !exact(authority_.frames_[frame], binding, state) ||
        authority_.frames_[frame].alias_claims != 0u ||
        !authority_.frames_[frame].dirty.empty() ||
        (authority_.frames_[frame].state != FrameState::Resident &&
         authority_.frames_[frame].state != FrameState::Empty)) {
      return RegistrationResult::Busy;
    }
    regions[index] = from_credential(binding).region;
  }
  if (!release_detail::ReleaseCheck::allowed(
          authority_, {regions.data(), state->binding_count()})) {
    return RegistrationResult::Busy;
  }
  for (std::size_t index = 0u; index < state->binding_count(); ++index) {
    authority_.frames_[regions[index].first] = Authority::Frame{};
  }
  state->set(registration_detail::Lifecycle::Released);
  return RegistrationResult::Done;
}

} // namespace rund::compute::detail::residency
