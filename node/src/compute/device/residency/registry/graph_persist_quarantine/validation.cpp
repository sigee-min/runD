#include "internal.hpp"

#include "../../execution/graph_persist/internal.hpp"
#include "../../registry.hpp"
#include "../frame.hpp"
#include "../release_check.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>

namespace rund::compute::detail::residency {

bool registry_model::CpuQuarantineOwner::check_epoch_q(
    const BookSnapshot &snap, const std::size_t index,
    Plan &plan) const noexcept {
  const BookSnapshot::Epoch &item = snap.epochs[index];
  const std::size_t expected_bank =
      index / graph_reduce::CpuReceiptBook::RoleCount;
  const std::uint8_t expected_role = static_cast<std::uint8_t>(
      index % graph_reduce::CpuReceiptBook::RoleCount);
  if (item.bank != expected_bank || item.role != expected_role) {
    return false;
  }
  const auto free = item.state == graph_reduce::CpuReceiptBook::SlotState::Free;
  if (free) {
    return (item.authority == nullptr || item.authority == &authority_) &&
           !item.key && item.token == 0u && item.generation == 0u &&
           item.permit == nullptr;
  }
  if ((item.state != graph_reduce::CpuReceiptBook::SlotState::Prepared &&
       item.state != graph_reduce::CpuReceiptBook::SlotState::Armed) ||
      item.authority != &authority_ || item.key.domain() != snap.domain ||
      item.key.slot() != index + 1u || !item.key || item.token == 0u ||
      item.generation == 0u || item.permit != nullptr ||
      item.handle != nullptr || plan.epoch_count >= plan.epochs.size()) {
    return false;
  }
  for (std::size_t candidate = 0u; candidate < plan.epoch_count; ++candidate) {
    if (plan.epochs[candidate] == index) {
      return false;
    }
  }
  for (std::size_t candidate = 0u;
       candidate < authority_.cycle_state_.epochs.size(); ++candidate) {
    const registry_model::LeaseSlot &epoch =
        authority_.cycle_state_.epochs[candidate];
    if (epoch.state != registry_model::LeaseState::CpuQuarantined ||
        epoch.cpu_key != item.key || epoch.token != item.token ||
        epoch.generation != item.generation || !epoch.terminal) {
      continue;
    }
    if (epoch.undo_frames.size() != epoch.undo.size()) {
      return false;
    }
    for (std::size_t row = 0u; row < epoch.undo_frames.size(); ++row) {
      const std::uint32_t frame = epoch.undo_frames[row];
      if (frame >= authority_.frames_.size() ||
          authority_.frames_[frame].alias_claims != 0u) {
        return false;
      }
      for (std::size_t prior = 0u; prior < row; ++prior) {
        if (epoch.undo_frames[prior] == frame) {
          return false;
        }
      }
    }
    for (std::size_t row = 0u; row < epoch.bindings.size(); ++row) {
      const CacheBinding &binding = epoch.bindings[row];
      if (binding.frame >= authority_.frames_.size() ||
          authority_.frames_[binding.frame].alias_claims != 0u) {
        return false;
      }
      for (std::size_t prior = 0u; prior < row; ++prior) {
        if (epoch.bindings[prior].frame == binding.frame) {
          return false;
        }
      }
    }
    for (std::size_t row = 0u; row < epoch.relocation_frames.size(); ++row) {
      const std::uint32_t frame = epoch.relocation_frames[row];
      if (frame >= authority_.frames_.size() ||
          authority_.frames_[frame].alias_claims != 0u) {
        return false;
      }
      for (std::size_t prior = 0u; prior < row; ++prior) {
        if (epoch.relocation_frames[prior] == frame) {
          return false;
        }
      }
    }
    plan.epochs[plan.epoch_count++] = candidate;
    return true;
  }
  return false;
}

bool registry_model::CpuQuarantineOwner::check_unknown_q(
    const BookSnapshot &snap, const std::size_t index,
    Plan &plan) const noexcept {
  const BookSnapshot::Unknown &item = snap.unknowns[index];
  if (item.authority != &authority_ || !item.terminal || !item.quarantined ||
      !item.live || item.domain != snap.domain || item.domain == 0u ||
      item.token == 0u || item.generation == 0u || item.coordinate == 0u ||
      item.plan == Identity{} || !item.identity.valid() ||
      item.page_count == 0u ||
      item.page_count > execution::GraphPersistCapacity ||
      plan.unknown_count >= plan.unknowns.size()) {
    return false;
  }
  const auto found = std::find_if(
      authority_.cycle_state_.graph_persists.begin(),
      authority_.cycle_state_.graph_persists.end(),
      [&](const registry_model::LeaseSlot &row) {
        return row.state == registry_model::LeaseState::Drain && row.terminal &&
               row.token == item.token && row.generation == item.generation &&
               row.coordinate == item.coordinate && row.plan == item.plan;
      });
  if (found == authority_.cycle_state_.graph_persists.end() ||
      found->book_domain != item.domain || found->identity != item.identity ||
      found->retry_region != item.region ||
      found->transitions.size() != item.page_count ||
      found->undo_frames.size() != item.page_count ||
      found->undo.size() != item.page_count || item.region.count == 0u ||
      item.region.first > authority_.frames_.size() ||
      item.region.count > authority_.frames_.size() - item.region.first) {
    return false;
  }
  if (!graph_persist_detail::check_undo(
          authority_.frames_, *found, graph_persist_detail::UndoMode::Drain)) {
    return false;
  }
  const std::size_t row_index = static_cast<std::size_t>(
      std::distance(authority_.cycle_state_.graph_persists.begin(), found));
  for (std::size_t prior = 0u; prior < plan.unknown_count; ++prior) {
    if (plan.unknowns[prior] == row_index) {
      return false;
    }
  }
  for (const CacheTransition &transition : found->transitions) {
    if (plan.claim_count >= plan.claims.size()) {
      return false;
    }
    for (std::size_t prior = 0u; prior < plan.claim_count; ++prior) {
      if (plan.claims[prior] == transition.frame) {
        return false;
      }
    }
    plan.claims[plan.claim_count++] = transition.frame;
  }
  plan.unknowns[plan.unknown_count++] = row_index;
  return true;
}

bool registry_model::CpuQuarantineOwner::check_retry_q(
    const BookSnapshot &snap, const std::size_t index,
    Plan &plan) const noexcept {
  const registry_model::LeaseSlot &row =
      authority_.cycle_state_.graph_persists[index];
  if (row.state != registry_model::LeaseState::RetryReady ||
      row.book_domain != snap.domain || row.book_domain == 0u ||
      row.token == 0u || row.generation == 0u || row.coordinate == 0u ||
      row.plan == Identity{} || row.terminal || !row.identity.valid() ||
      row.retry_region.count == 0u ||
      row.retry_region.tier != FrameTier::Host ||
      row.retry_region.role != FrameRole::Output ||
      row.retry_region.first > authority_.frames_.size() ||
      row.retry_region.count >
          authority_.frames_.size() - row.retry_region.first ||
      row.transitions.empty() ||
      row.transitions.size() != row.undo_frames.size() ||
      row.transitions.size() != row.undo.size() ||
      plan.retry_count >= plan.retries.size()) {
    return false;
  }
  if (!graph_persist_detail::check_undo(
          authority_.frames_, row, graph_persist_detail::UndoMode::Retry)) {
    return false;
  }
  for (std::size_t prior = 0u; prior < plan.retry_count; ++prior) {
    if (plan.retries[prior] == index) {
      return false;
    }
  }
  for (const CacheTransition &transition : row.transitions) {
    if (plan.claim_count >= plan.claims.size()) {
      return false;
    }
    for (std::size_t prior = 0u; prior < plan.claim_count; ++prior) {
      if (plan.claims[prior] == transition.frame) {
        return false;
      }
    }
    plan.claims[plan.claim_count++] = transition.frame;
  }
  plan.retries[plan.retry_count++] = index;
  return true;
}

bool registry_model::CpuQuarantineOwner::check_q(const BookSnapshot &snap,
                                                 Plan &plan) const noexcept {
  if (!snap.valid || snap.domain == 0u ||
      snap.live_receipts > snap.epochs.size() ||
      snap.live_permits > snap.epochs.size()) {
    return false;
  }
  std::size_t receipts = 0u;
  std::size_t permits = 0u;
  for (std::size_t index = 0u; index < snap.epochs.size(); ++index) {
    receipts += snap.epochs[index].handle != nullptr ? 1u : 0u;
    permits += snap.epochs[index].permit != nullptr ? 1u : 0u;
    if (!check_epoch_q(snap, index, plan)) {
      return false;
    }
  }
  if (receipts != snap.live_receipts || permits != snap.live_permits ||
      permits != 0u) {
    return false;
  }
  for (std::size_t index = 0u; index < snap.unknowns.size(); ++index) {
    const BookSnapshot::Unknown &item = snap.unknowns[index];
    if (!item.live) {
      if (item.authority != nullptr || item.plan != Identity{} ||
          item.identity != GraphPersistIdentity{} ||
          item.region != FrameRegion{} || item.page_count != 0u ||
          item.domain != 0u || item.token != 0u || item.generation != 0u ||
          item.coordinate != 0u || item.terminal || item.quarantined) {
        return false;
      }
      continue;
    }
    if (!check_unknown_q(snap, index, plan)) {
      return false;
    }
  }
  for (std::size_t index = 0u;
       index < authority_.cycle_state_.graph_persists.size(); ++index) {
    const registry_model::LeaseSlot &row =
        authority_.cycle_state_.graph_persists[index];
    if (row.state == registry_model::LeaseState::RetryReady) {
      if (!check_retry_q(snap, index, plan)) {
        return false;
      }
      continue;
    }
    if (row.state == registry_model::LeaseState::Free) {
      if (!release_detail::ReleaseCheck::idle(row)) {
        return false;
      }
      continue;
    }
    if (row.state != registry_model::LeaseState::Drain || !row.terminal) {
      return false;
    }
    bool found = false;
    for (std::size_t item = 0u; item < plan.unknown_count; ++item) {
      if (plan.unknowns[item] == index) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

} // namespace rund::compute::detail::residency
