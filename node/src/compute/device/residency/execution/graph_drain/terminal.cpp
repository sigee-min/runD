#include "../../registry/graph_drain_owner.hpp"
#include "internal.hpp"

namespace rund::compute::detail::residency {

bool GraphDrainOwner::terminal_graph_drain(
    execution::GraphDrain &ticket, const Status status,
    const execution::TerminalKind terminal, const bool may_write,
    const std::span<const execution::GraphDrainCompletion>
        completions) noexcept {
  if (!ticket || ticket.owner_ != &authority_ || ticket.terminalled_ ||
      ticket.page_count_ == 0u ||
      ticket.page_count_ > execution::GraphDrainCapacity ||
      completions.size() != ticket.page_count_) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  const AuthorityResult destination =
      authority_.resume_locked(ticket.destination_token_);
  if (!destination || destination.lease.bindings.size() != ticket.page_count_) {
    return false;
  }
  if (authority_.cycle_state_.writeback.token != ticket.source_token_ ||
      authority_.cycle_state_.writeback.state !=
          registry_model::LeaseState::Drain ||
      authority_.cycle_state_.writeback.transitions.size() !=
          ticket.page_count_ ||
      authority_.cycle_state_.writeback.undo_frames.size() !=
          ticket.page_count_ ||
      authority_.cycle_state_.writeback.undo.size() != ticket.page_count_) {
    return false;
  }
  bool exact = true;
  for (std::size_t index = 0u; index < ticket.page_count_; ++index) {
    const execution::GraphDrainPage page = ticket.pages_[index];
    const execution::GraphDrainCompletion completion = completions[index];
    const CacheBinding binding = destination.lease.bindings[index];
    const CacheTransition transition =
        authority_.cycle_state_.writeback.transitions[index];
    if (transition.kind != TransitionKind::Migrate ||
        transition.key != page.key || transition.frame != page.source_frame ||
        authority_.cycle_state_.writeback.undo_frames[index] !=
            page.source_frame ||
        transition.dirty.empty() ||
        page.source_frame >= authority_.frames_.size() ||
        authority_.frames_[page.source_frame].state != FrameState::Writeback ||
        authority_.frames_[page.source_frame].key != page.key ||
        binding.frame >= authority_.frames_.size() ||
        authority_.frames_[binding.frame].state != FrameState::Pinned ||
        authority_.frames_[binding.frame].key != page.key) {
      return false;
    }
    exact = exact && completion.key == page.key &&
            completion.source_frame == page.source_frame &&
            completion.target_frame == page.target_frame &&
            binding.key == page.key && binding.frame == page.target_frame &&
            binding.access == Access::Write &&
            (status ? completion.bytes == page.bytes
                    : completion.bytes <= page.bytes);
  }
  const bool contradictory =
      terminal == execution::TerminalKind::UnknownMayWrite
          ? static_cast<bool>(status)
          : false;
  ticket.completion_ = exact && !contradictory
                           ? status
                           : Status::fail(Reason::CompletionInvalid);
  ticket.terminal_ = terminal;
  ticket.completion_may_write_ =
      may_write || !exact || contradictory ||
      terminal == execution::TerminalKind::UnknownMayWrite;
  ticket.terminalled_ = true;
  return true;
}

} // namespace rund::compute::detail::residency
