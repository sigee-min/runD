#include "registration.hpp"

#include "../../../../../accel/kernel/residency/service_free_direct/validation.hpp"

#include <algorithm>
#include <new>
#include <utility>

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] FrameRole role_for(const std::uint32_t usage) noexcept {
  constexpr std::uint32_t read = rund::kernel::kResidentUsageRead;
  constexpr std::uint32_t write = rund::kernel::kResidentUsageWrite;
  if (usage == read) {
    return FrameRole::Input;
  }
  return usage == write ? FrameRole::Output : FrameRole::Intermediate;
}

[[nodiscard]] ResidentRecurrenceView
view_for(const rund::kernel::ResidentBufferRef &state) noexcept {
  return ResidentRecurrenceView{
      .resource = state.id,
      .bytes = state.bytes,
      .offset_bytes = state.offset_bytes,
      .element_bytes = state.element_bytes,
      .stride_bytes = state.stride_bytes,
      .count = state.count,
      .usage = state.usage,
  };
}

[[nodiscard]] registration_detail::State::Binding
credential_for(const ResidentRecurrenceBinding &binding) noexcept {
  return registration_detail::State::Binding{
      .resource = binding.view.resource,
      .bytes = binding.view.bytes,
      .offset_bytes = binding.view.offset_bytes,
      .element_bytes = binding.view.element_bytes,
      .stride_bytes = binding.view.stride_bytes,
      .count = binding.view.count,
      .usage = binding.view.usage,
      .region_first = binding.region.first,
      .region_count = binding.region.count,
      .tier = static_cast<std::uint8_t>(binding.region.tier),
      .role = static_cast<std::uint8_t>(binding.region.role),
      .registration = binding.registration};
}

[[nodiscard]] ResidentRecurrenceBinding
binding_for(const registration_detail::State::Binding &binding) noexcept {
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

DirectRecurrenceRegistration::~DirectRecurrenceRegistration() {
  static_cast<void>(release());
}

DirectRecurrenceRequest DirectRecurrenceRegistration::request() const noexcept {
  const std::shared_ptr<const DirectRecurrenceRegistration> owner =
      weak_from_this().lock();
  std::lock_guard lock{gate_};
  if (owner == nullptr || proof_ == nullptr || count_ != proof_->state_count ||
      count_ == 0u || registration_state_ == nullptr) {
    return {};
  }
  return DirectRecurrenceRequest{
      .proof_owner = std::static_pointer_cast<const void>(owner),
      .registration_state = registration_state_,
      .proof_hi = proof_->identity.hi,
      .proof_lo = proof_->identity.lo,
      .iterations = proof_->iterations,
  };
}

DirectRecurrenceRegistration::Snapshot
DirectRecurrenceRegistration::snapshot() const noexcept {
  std::lock_guard lock{gate_};
  Snapshot result{};
  result.proof = proof_;
  result.state = registration_state_;
  result.count = registration_state_ == nullptr
                     ? count_
                     : registration_state_->binding_count();
  if (registration_state_ == nullptr) {
    std::copy_n(bindings_.begin(), count_, result.bindings.begin());
  } else {
    for (std::size_t index = 0u; index < result.count; ++index) {
      result.bindings[index] = binding_for(registration_state_->binding(index));
    }
  }
  return result;
}

RegistrationResult DirectRecurrenceRegistration::rollback() noexcept {
  std::shared_ptr<Registry> registry;
  std::array<ResidentRecurrenceBinding, Capacity> bindings{};
  std::size_t count = 0u;
  {
    std::lock_guard lock{gate_};
    registry = registry_.lock();
    count = count_;
    std::copy_n(bindings_.begin(), count, bindings.begin());
  }
  if (registry == nullptr || count == 0u) {
    return registry == nullptr ? RegistrationResult::Invalid
                               : RegistrationResult::Done;
  }
  const RegistrationResult result =
      registry->authority().rollback_direct_registration(
          {bindings.data(), count});
  if (result == RegistrationResult::Done) {
    std::lock_guard lock{gate_};
    count_ = 0u;
    proof_.reset();
    registry_.reset();
    registration_state_.reset();
  }
  return result;
}

RegistrationResult DirectRecurrenceRegistration::release_pending(
    const DirectRecurrenceLease &lease) noexcept {
  std::shared_ptr<Registry> registry;
  std::shared_ptr<const registration_detail::State> registration_state;
  {
    std::lock_guard lock{gate_};
    registry = registry_.lock();
    registration_state = registration_state_;
  }
  if (registry == nullptr || registration_state == nullptr) {
    return RegistrationResult::Invalid;
  }
  return registry->authority()
      .direct_recurrences()
      .release_direct_recurrence_pending(lease, registration_state);
}

ExecutionClose DirectRecurrenceRegistration::finish_pending(
    const DirectRecurrenceLease &lease, void *const user,
    const DirectRecurrencePublication publication) noexcept {
  std::shared_ptr<Registry> registry;
  std::shared_ptr<const registration_detail::State> registration_state;
  {
    std::lock_guard lock{gate_};
    registry = registry_.lock();
    registration_state = registration_state_;
  }
  if (registry == nullptr) {
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }
  const ExecutionClose result =
      registry->authority().direct_recurrences().finish_direct_recurrence(
          lease, registration_state, user, publication);
  if (result.registration == RegistrationResult::Done && !result.quarantined) {
    std::lock_guard lock{gate_};
    count_ = 0u;
    proof_.reset();
    registry_.reset();
    registration_state_.reset();
  }
  return result;
}

RegistrationResult DirectRecurrenceRegistration::release() noexcept {
  std::shared_ptr<Registry> registry;
  std::shared_ptr<const registration_detail::State> state;
  {
    std::lock_guard lock{gate_};
    if (count_ == 0u) {
      return RegistrationResult::Done;
    }
    registry = registry_.lock();
    state = registration_state_;
  }
  if (registry == nullptr || state == nullptr) {
    return rollback();
  }
  const RegistrationResult result =
      registry->authority().direct_recurrences().retire_direct_registration(
          state);
  if (result == RegistrationResult::Done) {
    std::lock_guard lock{gate_};
    count_ = 0u;
    proof_.reset();
    registry_.reset();
    registration_state_.reset();
  }
  return result;
}

std::shared_ptr<DirectRecurrenceRegistration> register_direct_recurrence_proof(
    const std::shared_ptr<Registry> &registry,
    std::shared_ptr<const node::accel::detail::ServiceFreeDirectProof>
        proof) noexcept {
  if (registry == nullptr || proof == nullptr ||
      !node::accel::detail::service_free_direct_proof_valid(*proof)) {
    return {};
  }
  try {
    auto owner = std::shared_ptr<DirectRecurrenceRegistration>(
        new DirectRecurrenceRegistration());
    owner->registry_ = registry;
    owner->proof_ = std::move(proof);
    std::array<registration_detail::State::Binding,
               DirectRecurrenceRegistration::Capacity>
        credentials{};
    for (std::size_t index = 0u; index < owner->proof_->state_count; ++index) {
      const rund::kernel::ResidentBufferRef state =
          owner->proof_->states[index];
      ResidentRecurrenceBinding binding{};
      if (!registry->authority().register_resident_recurrence_view(
              role_for(state.usage), view_for(state), binding)) {
        const RegistrationResult result = owner->rollback();
        return result == RegistrationResult::Done ? nullptr : owner;
      }
      owner->bindings_[owner->count_] = binding;
      credentials[owner->count_] = credential_for(binding);
      ++owner->count_;
    }
    const std::shared_ptr<const void> sealed_owner =
        std::static_pointer_cast<const void>(owner);
    owner->registration_state_ = registration_detail::make(
        registration_detail::Policy::Release, sealed_owner,
        {credentials.data(), owner->count_});
    if (owner->registration_state_ == nullptr ||
        !registry->authority().direct_recurrences().bind_direct_registration(
            owner->registration_state_)) {
      owner->registration_state_.reset();
      const RegistrationResult result = owner->rollback();
      return result == RegistrationResult::Done ? nullptr : owner;
    }
    return owner;
  } catch (const std::bad_alloc &) {
    return {};
  }
}

} // namespace rund::compute::detail::residency
