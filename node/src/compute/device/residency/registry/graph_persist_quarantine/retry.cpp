#include "internal.hpp"

#include "../../execution/graph_persist/internal.hpp"
#include "../../registry.hpp"
#include "../frame.hpp"
#include "../graph_persist_owner.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <limits>
#include <mutex>

namespace rund::compute::detail::residency {

bool GraphPersistOwner::validate_cpu_graph_persist(
    const execution::GraphPersist &ticket) const noexcept {
  if (!ticket || ticket.owner_ != &authority_ || ticket.token_ == 0u ||
      ticket.generation_ == 0u || ticket.coordinate_ == 0u ||
      ticket.plan_ == Identity{} || ticket.book_domain_ == 0u ||
      !ticket.identity().valid() || ticket.page_count_ == 0u ||
      ticket.page_count_ > execution::GraphPersistCapacity) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  const auto found =
      std::find_if(authority_.cycle_state_.graph_persists.begin(),
                   authority_.cycle_state_.graph_persists.end(),
                   [&](const Authority::LeaseSlot &slot) {
                     return slot.token == ticket.token_ &&
                            slot.state == Authority::LeaseState::Drain;
                   });
  if (found == authority_.cycle_state_.graph_persists.end() ||
      !graph_persist_detail::matches(*found, ticket) ||
      !graph_persist_detail::check_undo(
          authority_.frames_, *found, graph_persist_detail::UndoMode::Drain)) {
    return false;
  }
  return true;
}

bool GraphPersistOwner::has_foreign_cpu_graph_retry(
    const std::uint64_t domain) const noexcept {
  std::lock_guard lock{authority_.gate_};
  return std::any_of(authority_.cycle_state_.graph_persists.begin(),
                     authority_.cycle_state_.graph_persists.end(),
                     [domain](const Authority::LeaseSlot &slot) {
                       return slot.state == Authority::LeaseState::RetryReady &&
                              slot.book_domain != domain;
                     });
}

bool GraphPersistOwner::cpu_graph_retry_active() const noexcept {
  std::lock_guard lock{authority_.gate_};
  return std::any_of(authority_.cycle_state_.graph_persists.begin(),
                     authority_.cycle_state_.graph_persists.end(),
                     [](const Authority::LeaseSlot &slot) {
                       return slot.state == Authority::LeaseState::RetryReady;
                     });
}

bool GraphPersistOwner::has_cpu_graph_retry(
    const std::uint64_t domain) const noexcept {
  if (domain == 0u || domain == std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  return std::any_of(authority_.cycle_state_.graph_persists.begin(),
                     authority_.cycle_state_.graph_persists.end(),
                     [domain](const Authority::LeaseSlot &slot) {
                       return slot.state == Authority::LeaseState::RetryReady &&
                              slot.book_domain == domain;
                     });
}

bool GraphPersistOwner::discard_cpu_graph_retry(
    const std::uint64_t domain) noexcept {
  if (domain == 0u || domain == std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  std::array<std::uint32_t, execution::GraphPersistAggregateCapacity> frames{};
  std::array<std::size_t, execution::GraphPersistSlotCapacity> rows{};
  std::size_t row_count = 0u;
  std::size_t frame_count = 0u;
  for (std::size_t index = 0u;
       index < authority_.cycle_state_.graph_persists.size(); ++index) {
    const Authority::LeaseSlot &slot =
        authority_.cycle_state_.graph_persists[index];
    if (slot.state != Authority::LeaseState::RetryReady) {
      continue;
    }
    if (slot.book_domain != domain || slot.token == 0u ||
        slot.generation == 0u || slot.coordinate == 0u ||
        slot.plan == Identity{} || slot.terminal || !slot.identity.valid() ||
        row_count >= rows.size() || slot.retry_region.count == 0u ||
        slot.retry_region.first > authority_.frames_.size() ||
        slot.retry_region.count >
            authority_.frames_.size() - slot.retry_region.first ||
        slot.transitions.empty() ||
        slot.transitions.size() != slot.undo_frames.size() ||
        slot.transitions.size() != slot.undo.size() ||
        slot.transitions.size() > frames.size() - frame_count) {
      return false;
    }
    if (!graph_persist_detail::check_undo(
            authority_.frames_, slot, graph_persist_detail::UndoMode::Retry)) {
      return false;
    }
    for (const CacheTransition &transition : slot.transitions) {
      if (frame_count >= frames.size()) {
        return false;
      }
      for (std::size_t prior = 0u; prior < frame_count; ++prior) {
        if (frames[prior] == transition.frame) {
          return false;
        }
      }
      frames[frame_count++] = transition.frame;
    }
    rows[row_count++] = index;
  }
  for (std::size_t index = 0u; index < frame_count; ++index) {
    authority_.frames_[frames[index]] =
        frame_detail::empty(authority_.frames_[frames[index]]);
  }
  for (std::size_t index = 0u; index < row_count; ++index) {
    ::rund::compute::detail::residency::clear(
        authority_.cycle_state_.graph_persists[rows[index]]);
  }
  return true;
}

registry_model::RetryTxn::~RetryTxn() noexcept {
  if (active) {
    rollback();
  }
}

bool registry_model::RetryTxn::stage(
    const std::uint64_t domain, const GraphPersistIdentity &expected) noexcept {
  if (domain == 0u || domain == std::numeric_limits<std::uint64_t>::max() ||
      !expected.valid()) {
    return false;
  }
  for (std::size_t index = 0u; index < owner.cycle_state_.graph_persists.size();
       ++index) {
    registry_model::LeaseSlot &row = owner.cycle_state_.graph_persists[index];
    if (row.state != registry_model::LeaseState::RetryReady) {
      continue;
    }
    if (row.book_domain != domain || row.token == 0u || row.generation == 0u ||
        row.coordinate == 0u || row.plan == Identity{} || row.terminal ||
        !row.identity.valid() || row.identity != expected ||
        row.retry_region.count == 0u ||
        row.retry_region.first > owner.frames_.size() ||
        row.retry_region.count >
            owner.frames_.size() - row.retry_region.first ||
        row.transitions.empty() ||
        row.transitions.size() != row.undo_frames.size() ||
        row.transitions.size() != row.undo.size() || row_count >= rows.size() ||
        row.transitions.size() > frames.size() - frame_count) {
      return false;
    }
    if (!graph_persist_detail::check_undo(
            owner.frames_, row, graph_persist_detail::UndoMode::Retry)) {
      return false;
    }
    for (const CacheTransition &transition : row.transitions) {
      if (frame_count >= frames.size()) {
        return false;
      }
      for (std::size_t prior = 0u; prior < frame_count; ++prior) {
        if (frames[prior] == transition.frame) {
          return false;
        }
      }
      frames[frame_count++] = transition.frame;
    }
    rows[row_count++] = index;
  }
  if (row_count == 0u) {
    return true;
  }
  for (std::size_t index = 0u; index < frame_count; ++index) {
    owner.frames_[frames[index]] =
        frame_detail::empty(owner.frames_[frames[index]]);
  }
  for (std::size_t index = 0u; index < row_count; ++index) {
    owner.cycle_state_.graph_persists[rows[index]].state =
        registry_model::LeaseState::Computing;
  }
  active = true;
  return true;
}

void registry_model::RetryTxn::commit() noexcept {
  if (!active) {
    return;
  }
  for (std::size_t index = 0u; index < row_count; ++index) {
    ::rund::compute::detail::residency::clear(
        owner.cycle_state_.graph_persists[rows[index]]);
  }
  active = false;
}

void registry_model::RetryTxn::rollback() noexcept {
  if (!active) {
    return;
  }
  for (std::size_t index = 0u; index < row_count; ++index) {
    registry_model::LeaseSlot &row =
        owner.cycle_state_.graph_persists[rows[index]];
    if (!graph_persist_detail::check_undo(
            owner.frames_, row, graph_persist_detail::UndoMode::Staged)) {
      std::terminate();
    }
    graph_persist_detail::rollback(owner.frames_, row);
    row.state = registry_model::LeaseState::RetryReady;
  }
  active = false;
}

} // namespace rund::compute::detail::residency
