#include "cpu_graph_owner.hpp"

#include "graph_persist_quarantine/internal.hpp"
#include "internal.hpp"

#include "../execution/graph_persist/internal.hpp"
#include "frame.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

namespace rund::compute::detail::residency {

using Frame = registry_model::Frame;
using LeaseSlot = registry_model::LeaseSlot;
using LeaseState = registry_model::LeaseState;

CpuGraphOwner::CpuGraphOwner(Authority &authority) noexcept
    : authority_(authority) {}

CpuGraphOwner Authority::cpu_graph() noexcept { return CpuGraphOwner{*this}; }

std::uint64_t CpuGraphOwner::cpu_owner_id() const noexcept {
  return authority_.credentials_.owner_id;
}

CpuReservationKey CpuGraphOwner::reserve_cpu_key(
    const std::uint64_t domain, const std::uint64_t role_slot,
    const std::uint64_t nonce) noexcept {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (domain == 0u || domain == max || role_slot >= max - 1u || nonce == 0u ||
      nonce == max || authority_.credentials_.owner_id == 0u ||
      authority_.credentials_.owner_id == max) {
    return {};
  }
  const CpuReservationKey key = CpuReservationKey::make(
      domain, role_slot + 1u, nonce, authority_.credentials_.owner_id);
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.cpu_graph_state_.pending_cpu ||
      std::any_of(authority_.cycle_state_.epochs.begin(),
                  authority_.cycle_state_.epochs.end(),
                  [key](const LeaseSlot &slot) {
                    return slot.state != LeaseState::Free &&
                           slot.cpu_key == key;
                  })) {
    return {};
  }
  authority_.cpu_graph_state_.pending_cpu = key;
  return key;
}

bool CpuGraphOwner::cancel_cpu_reservation(
    const CpuReservationKey key) noexcept {
  if (!valid_cpu_key(key) ||
      key.owner() != authority_.credentials_.owner_id) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  if (authority_.cpu_graph_state_.pending_cpu == key) {
    authority_.cpu_graph_state_.pending_cpu = {};
    return true;
  }
  if (authority_.cpu_graph_state_.pending_cpu) {
    return false;
  }
  return std::none_of(
      authority_.cycle_state_.epochs.begin(), authority_.cycle_state_.epochs.end(),
      [key](const LeaseSlot &slot) {
        return slot.state != LeaseState::Free && slot.cpu_key == key;
      });
}

bool CpuGraphOwner::confirm_cpu_epoch(const CpuReservationKey key,
                                      const std::uint64_t token,
                                      const std::uint64_t generation) noexcept {
  if (!valid_cpu_key(key) ||
      key.owner() != authority_.credentials_.owner_id || token == 0u ||
      generation == 0u) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  const auto epoch = std::find_if(
      authority_.cycle_state_.epochs.begin(), authority_.cycle_state_.epochs.end(),
      [token](const LeaseSlot &slot) { return slot.token == token; });
  if (epoch == authority_.cycle_state_.epochs.end() || epoch->cpu_key != key ||
      epoch->generation != generation || epoch->state != LeaseState::Prepared ||
      epoch->cpu_bound || epoch->cycle != 0u) {
    return false;
  }
  epoch->cpu_bound = true;
  return true;
}

bool CpuGraphOwner::close_cpu_epoch(const CpuReservationKey key,
                                    const std::uint64_t token,
                                    const std::uint64_t generation,
                                    const bool success,
                                    const bool invalidate_all,
                                    CloseInfo *const info) noexcept {
  if (info != nullptr) {
    *info = CloseInfo{};
    info->token = token;
    info->generation = generation;
    info->flags = static_cast<std::uint8_t>(
        (success ? CloseInfo::Success : CloseInfo::Flag{}) |
        (invalidate_all ? CloseInfo::Invalidate : CloseInfo::Flag{}));
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    if (info != nullptr) {
      info->check = CloseInfo::Check::State;
    }
    return false;
  }
  if (!valid_cpu_key(key) ||
      key.owner() != authority_.credentials_.owner_id) {
    if (info != nullptr) {
      info->check = CloseInfo::Check::Credential;
    }
    return false;
  }
  if (token == 0u || generation == 0u) {
    if (success || !invalidate_all) {
      if (info != nullptr) {
        info->check =
            success ? CloseInfo::Check::State : CloseInfo::Check::Invalidate;
      }
      return false;
    }
    const auto malformed = std::find_if(
        authority_.cycle_state_.epochs.begin(), authority_.cycle_state_.epochs.end(),
        [key](const LeaseSlot &slot) {
          return slot.state != LeaseState::Free && slot.cpu_key == key;
        });
    if (malformed == authority_.cycle_state_.epochs.end()) {
      if (info != nullptr) {
        info->check = CloseInfo::Check::Credential;
      }
      return false;
    }
    if (info != nullptr) {
      info->token = malformed->token;
      info->generation = malformed->generation;
    }
    if (complete_epoch(authority_.frames_, *malformed, false, true, info)) {
      return true;
    }
    malformed->state = LeaseState::CpuQuarantined;
    malformed->terminal = true;
    return false;
  }
  const auto epoch = std::find_if(
      authority_.cycle_state_.epochs.begin(), authority_.cycle_state_.epochs.end(),
      [token](const LeaseSlot &slot) { return slot.token == token; });
  if (epoch == authority_.cycle_state_.epochs.end() || epoch->cpu_key != key ||
      epoch->generation != generation || epoch->cycle != 0u) {
    if (info != nullptr) {
      info->check = CloseInfo::Check::Credential;
    }
    return false;
  }
  if (success && !epoch->cpu_bound) {
    if (info != nullptr) {
      info->check = CloseInfo::Check::State;
    }
    return false;
  }
  const auto witness = [&](
      const std::uint32_t frame,
      const std::size_t binding_index =
          std::numeric_limits<std::size_t>::max(),
      const CacheBinding *const binding = nullptr,
      const CloseInfo::Check check = CloseInfo::Check::Alias) noexcept {
    if (info == nullptr) {
      return;
    }
    info->check = check;
    info->token = token;
    info->generation = generation;
    info->frame_index = frame;
    if (binding != nullptr) {
      info->binding_index =
          binding_index == std::numeric_limits<std::size_t>::max()
              ? CloseInfo::NoIndex
              : static_cast<std::uint32_t>(binding_index);
      info->binding_key = binding->key;
      info->binding_access = binding->access;
      info->binding_dirty = binding->dirty;
      info->binding_prior_dirty = binding->prior_dirty;
      info->binding_next_use = binding->next_use;
      info->binding_retain_until = binding->retain_until;
      info->binding_fetch = binding->fetch;
      info->binding_relocated = binding->relocated;
      info->binding_retire = binding->retire_on_success;
      info->flags = static_cast<std::uint8_t>(
          info->flags |
          (binding->fetch ? CloseInfo::Fetch : CloseInfo::Flag{}) |
          (binding->relocated ? CloseInfo::Relocated : CloseInfo::Flag{}) |
          (binding->retire_on_success ? CloseInfo::Retire : CloseInfo::Flag{}));
    }
    if (frame < authority_.frames_.size()) {
      const Frame &value = authority_.frames_[frame];
      info->frame_key = value.key;
      info->frame_assigned = value.assigned;
      info->frame_state = value.state;
      info->frame_tier = value.tier;
      info->frame_role = value.role;
      info->frame_dirty = value.dirty;
      info->extent = value.extent;
      info->view = value.view;
      info->claims = value.alias_claims;
      info->frame_next_use = value.next_use;
      info->frame_retain_until = value.retain_until;
    }
  };
  if (epoch->state == LeaseState::CpuQuarantined) {
    if (success || !invalidate_all || !epoch->terminal) {
      if (info != nullptr) {
        info->check = success ? CloseInfo::Check::State
                              : (!invalidate_all ? CloseInfo::Check::Invalidate
                                                 : CloseInfo::Check::State);
      }
      return false;
    }
    for (std::size_t index = 0u; index < epoch->bindings.size(); ++index) {
      const CacheBinding &binding = epoch->bindings[index];
      if (binding.frame >= authority_.frames_.size() ||
          authority_.frames_[binding.frame].alias_claims != 0u) {
        witness(binding.frame, index, &binding,
                binding.frame >= authority_.frames_.size()
                    ? CloseInfo::Check::Bounds
                    : CloseInfo::Check::Alias);
        return false;
      }
    }
    for (const std::uint32_t frame : epoch->undo_frames) {
      if (frame >= authority_.frames_.size() ||
          authority_.frames_[frame].alias_claims != 0u) {
        witness(frame, std::numeric_limits<std::size_t>::max(), nullptr,
                frame >= authority_.frames_.size() ? CloseInfo::Check::Bounds
                                                   : CloseInfo::Check::Alias);
        return false;
      }
    }
    for (const std::uint32_t frame : epoch->relocation_frames) {
      if (frame >= authority_.frames_.size() ||
          authority_.frames_[frame].alias_claims != 0u) {
        witness(frame, std::numeric_limits<std::size_t>::max(), nullptr,
                frame >= authority_.frames_.size()
                    ? CloseInfo::Check::Relocation
                    : CloseInfo::Check::Alias);
        return false;
      }
    }
  }
  if (complete_epoch(authority_.frames_, *epoch, success, invalidate_all,
                     info)) {
    return true;
  }
  epoch->state = LeaseState::CpuQuarantined;
  epoch->terminal = true;
  return false;
}

bool CpuGraphOwner::retain_cpu_quarantine(
    std::shared_ptr<graph_reduce::CpuGraphQuarantine> quarantine) noexcept {
  if (quarantine == nullptr ||
      quarantine->phase != graph_reduce::CpuGraphQuarantine::Phase::Armed ||
      quarantine->book == nullptr || quarantine->authority != &authority_ ||
      quarantine->pool == nullptr || quarantine->pipeline == nullptr) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.cpu_graph_state_.cpu_quarantine != nullptr) {
    return false;
  }
  authority_.cpu_graph_state_.cpu_quarantine = std::move(quarantine);
  return true;
}

bool CpuGraphOwner::discard_cpu_quarantine(
    const std::shared_ptr<graph_reduce::CpuGraphQuarantine> &quarantine,
    const void *const pipeline_identity,
    const void *const pool_identity) noexcept {
  if (quarantine == nullptr ||
      quarantine->phase != graph_reduce::CpuGraphQuarantine::Phase::Held ||
      quarantine->book == nullptr || quarantine->authority != &authority_ ||
      quarantine->pipeline != pipeline_identity ||
      quarantine->pool != pool_identity || pipeline_identity == nullptr ||
      pool_identity == nullptr) {
    return false;
  }
  graph_reduce::CpuReceiptBook &book = *quarantine->book;
  registry_model::CpuQuarantineOwner owner{authority_};
  if (!owner.discard(book, quarantine.get())) {
    return false;
  }
  quarantine->clear();
  return true;
}

bool CpuGraphOwner::cpu_quarantine_active() const noexcept {
  std::lock_guard lock{authority_.gate_};
  return authority_.cpu_graph_state_.cpu_quarantine != nullptr;
}

} // namespace rund::compute::detail::residency
