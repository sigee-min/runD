#include "../../registry/graph_persist_owner.hpp"
#include "internal.hpp"

namespace rund::compute::detail::residency {

bool GraphPersistOwner::terminal_graph_persist(
    execution::GraphPersist &ticket, const Status status,
    const execution::TerminalKind terminal, const bool may_write,
    const std::span<const execution::GraphPersistCompletion>
        completions) noexcept {
  if (!ticket || ticket.owner_ != &authority_ || ticket.terminalled_ ||
      completions.size() != ticket.page_count_) {
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
  // Authenticate the complete Authority row before touching any terminal
  // evidence.  In particular, a token/generation match is not enough: the
  // prepared identity and every page transition belong to the same row.
  if (found == authority_.cycle_state_.graph_persists.end() ||
      !graph_persist_detail::matches(*found, ticket)) {
    return false;
  }
  if (!graph_persist_detail::check_undo(
          authority_.frames_, *found, graph_persist_detail::UndoMode::Drain)) {
    return false;
  }
  bool exact = true;
  bool wrote = false;
  for (std::size_t index = 0u; index < ticket.page_count_; ++index) {
    const execution::GraphPersistPage page = ticket.pages_[index];
    const execution::GraphPersistCompletion completion = completions[index];
    const CacheTransition transition = found->transitions[index];
    exact = exact && completion.key == page.key &&
            completion.backing_offset == page.backing_offset &&
            completion.frame == page.frame && transition.key == page.key &&
            transition.frame == page.frame &&
            transition.kind == TransitionKind::Writeback &&
            transition.dirty.offset == page.backing_offset &&
            transition.dirty.bytes == page.bytes &&
            page.frame < authority_.frames_.size() &&
            authority_.frames_[page.frame].state == FrameState::Writeback &&
            authority_.frames_[page.frame].key == page.key &&
            (status ? completion.bytes == page.bytes
                    : completion.bytes <= page.bytes);
    wrote = wrote || completion.bytes != 0u;
  }
  const bool contradictory =
      terminal == execution::TerminalKind::UnknownMayWrite
          ? static_cast<bool>(status)
          : false;
  ticket.identity_bad_ = !exact || contradictory;
  ticket.completion_ = exact && !contradictory
                           ? status
                           : Status::fail(Reason::CompletionInvalid);
  ticket.terminal_ = terminal;
  ticket.completion_may_write_ =
      may_write || wrote || !exact || contradictory ||
      terminal == execution::TerminalKind::UnknownMayWrite;
  ticket.terminalled_ = true;
  return true;
}

} // namespace rund::compute::detail::residency
