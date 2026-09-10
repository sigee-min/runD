#include "../../registry/graph_persist_owner.hpp"
#include "internal.hpp"

namespace rund::compute::detail::residency {

bool GraphPersistOwner::close_graph_persist(execution::GraphPersist &ticket,
                                            const bool unknown) noexcept {
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
      !graph_persist_detail::matches(*found, ticket)) {
    return false;
  }
  Authority::LeaseSlot &slot = *found;
  if (!graph_persist_detail::check_undo(
          authority_.frames_, slot, graph_persist_detail::UndoMode::Drain)) {
    return false;
  }
  if (unknown || ticket.identity_bad_ || ticket.unknown()) {
    slot.terminal = true;
    ticket.terminal_ = execution::TerminalKind::UnknownMayWrite;
    ticket.completion_may_write_ = true;
    ticket.terminalled_ = true;
    if (ticket.book_domain_ == 0u) {
      ticket.clear();
      return true;
    }
    return false;
  }

  const bool success = static_cast<bool>(ticket.completion_);
  if (success) {
    if (!graph_persist_detail::finish(authority_.frames_, slot, true)) {
      slot.terminal = true;
      ticket.identity_bad_ = true;
      return false;
    }
    ticket.clear();
    return true;
  }

  if (ticket.book_domain_ == 0u) {
    if (!graph_persist_detail::finish(authority_.frames_, slot, false)) {
      return false;
    }
    ticket.clear();
    return true;
  }
  if (slot.book_domain == 0u || slot.retry_region != ticket.region_ ||
      slot.retry_region.count == 0u || slot.transitions.empty() ||
      slot.transitions.size() != slot.undo_frames.size() ||
      slot.transitions.size() != slot.undo.size()) {
    return false;
  }
  graph_persist_detail::rollback(authority_.frames_, slot);
  slot.state = Authority::LeaseState::RetryReady;
  slot.terminal = false;
  ticket.clear();
  return true;
}

bool GraphPersistOwner::release_graph_persist(
    execution::GraphPersist &ticket) noexcept {
  if (!ticket || ticket.owner_ != &authority_ || !ticket.terminalled_) {
    return false;
  }
  return close_graph_persist(ticket, false);
}

bool GraphPersistOwner::release_graph_persist(
    execution::GraphPersist &&ticket) noexcept {
  return release_graph_persist(ticket);
}

bool GraphPersistOwner::recover_cpu_graph_persist(
    execution::GraphPersist &ticket, const bool unknown) noexcept {
  if (!ticket || ticket.owner_ != &authority_ || ticket.token_ == 0u ||
      ticket.generation_ == 0u || ticket.coordinate_ == 0u ||
      ticket.plan_ == Identity{} || !ticket.identity().valid() ||
      ticket.page_count_ == 0u ||
      ticket.page_count_ > execution::GraphPersistCapacity) {
    return false;
  }
  return close_graph_persist(ticket, unknown);
}

} // namespace rund::compute::detail::residency
