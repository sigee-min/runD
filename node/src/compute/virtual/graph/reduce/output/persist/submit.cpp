#include "../persist.hpp"

#include <span>

namespace rund::compute::detail::graph_reduce {

Status PersistController::submit_async(Ticket &ticket,
                                       bool &child_poison) noexcept {
  output_persist_detail::Projection projection{};
  if (pool_.graph_persist == nullptr ||
      ticket.bank >= residency::Pool::BankCount ||
      !output_persist_detail::project(run_, ticket, projection) ||
      !pool_.graph_persist->slots[ticket.bank].submit(
          *output_,
          std::span<const residency::PersistRequest>{projection.requests.data(),
                                                     projection.count},
          projection.token)) {
    const bool cancelled = output_persist_detail::cancel_unsubmitted(
        authority_, ticket.output_persist);
    ticket.phase = TicketPhase::HostOutputDirty;
    child_poison = !cancelled || child_poison;
    return Status::fail(cancelled ? Reason::PipelineBusy
                                  : Reason::CompletionInvalid);
  }
  ticket.phase = TicketPhase::OutputPersisting;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
