#include "../persist.hpp"

#include "../../../../../device/residency/registry/graph_persist_owner.hpp"

#include <array>
#include <span>

namespace rund::compute::detail::graph_reduce {

Status PersistController::retire(Ticket &ticket, bool &child_poison) noexcept {
  if (output_ == nullptr || pool_.graph_persist == nullptr ||
      ticket.bank >= residency::Pool::BankCount ||
      ticket.phase != TicketPhase::OutputPersisting || !ticket.output_persist) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::PersistReceipt receipt =
      pool_.graph_persist->slots[ticket.bank].wait();
  const auto expected = ticket.output_persist.pages();
  std::array<residency::execution::GraphPersistCompletion,
             residency::execution::GraphPersistCapacity>
      completions{};
  std::uint64_t bytes = 0u;
  std::size_t completed_pages = 0u;
  bool exact = receipt.token == ticket.batch + 1u &&
               receipt.pages.size() == expected.size();
  for (std::size_t index = 0u;
       index < receipt.pages.size() && index < expected.size(); ++index) {
    const residency::PersistedPage page = receipt.pages[index];
    completions[index] = residency::execution::GraphPersistCompletion{
        .key = page.key,
        .backing_offset = page.backing_offset,
        .bytes = page.bytes,
        .frame = page.physical_frame,
    };
    exact = exact && page.key == expected[index].key &&
            page.backing_offset == expected[index].backing_offset &&
            page.physical_frame == expected[index].frame &&
            page.bytes <= expected[index].bytes;
    bytes += page.bytes;
    completed_pages += page.bytes != 0u ? 1u : 0u;
  }
  output_persist_detail::record(stats_, receipt.io_ns, bytes, completed_pages);

  const Status status =
      exact ? receipt.status : Status::fail(Reason::CompletionInvalid);
  auto persist_owner = authority_.graph_persists();
  const bool hashed = !status || output_persist_detail::hash_persisted(
                                     run_, ticket.output_persist, hash_);
  const bool terminal = persist_owner.terminal_graph_persist(
      ticket.output_persist,
      hashed ? status : Status::fail(Reason::CompletionInvalid),
      residency::execution::TerminalKind::Known,
      receipt.may_write || !exact || !hashed,
      std::span<const residency::execution::GraphPersistCompletion>{
          completions.data(), expected.size()});
  const bool released =
      terminal && persist_owner.release_graph_persist(ticket.output_persist);
  if (!terminal || !released || !hashed) {
    // Preserve the move-only GraphPersist for AbortController recovery and
    // make the one-shot terminal transition non-reentrant.
    ticket.phase = TicketPhase::HostOutputDirty;
    child_poison = true;
    return Status::fail(Reason::CompletionInvalid);
  }
  if (!status) {
    ticket.phase = TicketPhase::HostOutputDirty;
    return status;
  }
  ticket.output_dirty = false;
  return reset_ticket(ticket) ? Status::success()
                              : Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail::graph_reduce
