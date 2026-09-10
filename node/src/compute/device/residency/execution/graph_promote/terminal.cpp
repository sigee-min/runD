#include "internal.hpp"

namespace rund::compute::detail::residency {

bool GraphPromoteOwner::terminal_graph_promote(
    execution::GraphPromote &ticket, const Status status,
    const execution::TerminalKind terminal, const bool may_write,
    const std::span<const execution::GraphPromoteCompletion>
        completions) noexcept {
  if (!ticket || ticket.owner_ != &authority_ || ticket.terminalled_ ||
      completions.size() != ticket.page_count_) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  bool exact = true;
  bool wrote = false;
  for (std::size_t index = 0u; index < ticket.page_count_; ++index) {
    const execution::GraphPromotePage page = ticket.pages_[index];
    const execution::GraphPromoteCompletion completion = completions[index];
    exact = exact && completion.key == page.key &&
            completion.source_frame == page.source_frame &&
            completion.target_frame == page.target_frame &&
            (status ? completion.bytes == page.bytes
                    : completion.bytes <= page.bytes);
    wrote = wrote || completion.bytes != 0u;
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
      may_write || (!status && wrote) || !exact || contradictory ||
      terminal == execution::TerminalKind::UnknownMayWrite;
  ticket.terminalled_ = true;
  return true;
}

} // namespace rund::compute::detail::residency
