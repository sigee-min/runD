#include "../persist.hpp"

#include "../../../../../device/residency/registry/graph_persist_owner.hpp"

#include <span>

namespace rund::compute::detail::graph_reduce {

Status PersistController::persist_cpu(Ticket &ticket,
                                      bool &child_poison) noexcept {
  const output_persist_detail::PersistIo io = output_persist_detail::perform(
      *output_, run_, ticket.output_persist, hash_);
  output_persist_detail::record(stats_,
                                io.interval.completed - io.interval.started,
                                io.bytes, io.completed_pages);
  auto persist_owner = authority_.graph_persists();
  const bool terminal = persist_owner.terminal_graph_persist(
      ticket.output_persist, io.status,
      residency::execution::TerminalKind::Known, io.may_write,
      std::span<const residency::execution::GraphPersistCompletion>{
          io.completions.data(), io.page_count});
  const bool released = terminal && persist_owner.release_graph_persist(
                                        ticket.output_persist);
  if (!terminal || !released) {
    // Keep the armed credential visible to AbortController.  Marking the
    // ticket dirty also prevents PersistController::finish from attempting a
    // second terminal transition during whole-run cleanup.
    ticket.phase = TicketPhase::HostOutputDirty;
    child_poison = true;
    return Status::fail(Reason::CompletionInvalid);
  }
  if (!io.status) {
    ticket.phase = TicketPhase::HostOutputDirty;
    return io.status;
  }
  if (!wavefront_.release(ticket.batch)) {
    child_poison = true;
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.output_dirty = false;
  return reset_ticket(ticket) ? Status::success()
                              : Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail::graph_reduce
