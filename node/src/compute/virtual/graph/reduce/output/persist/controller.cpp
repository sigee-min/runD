#include "../persist.hpp"

#include "../../../../backing.hpp"

#include <rund/compute/virtual.hpp>

#include <algorithm>

namespace rund::compute::detail::graph_reduce {

PersistController::PersistController(
    residency::Authority &authority, residency::Pool &pool,
    const std::shared_ptr<const residency::ResidencyPlan> &owner,
    VirtualBacking *const output, const VirtualRunProjection &run,
    const residency::TiledGraphPlan &graph, Stats &stats, Wavefront &wavefront,
    StageController &stages, const std::size_t terminal_stage,
    ::rund::node::hash_detail::Fnv &hash) noexcept
    : authority_(authority), pool_(pool), owner_(owner), output_(output),
      run_(run), graph_(graph), stats_(stats), wavefront_(wavefront),
      stages_(stages), terminal_stage_(terminal_stage), hash_(hash),
      parallel_(output != nullptr &&
                VirtualBackingAccess::write_lanes(*output) >= 2u) {}

Status PersistController::before_start(const std::span<Ticket> tickets,
                                       Ticket &current,
                                       bool &child_poison) noexcept {
  if (output_ == nullptr || parallel_) {
    return Status::success();
  }
  for (Ticket &ticket : tickets) {
    if (&ticket != &current && ticket.phase == TicketPhase::OutputPersisting) {
      return retire(ticket, child_poison);
    }
  }
  return Status::success();
}

Status PersistController::reuse(Ticket &ticket, bool &child_poison) noexcept {
  return ticket.phase == TicketPhase::OutputPersisting
             ? retire(ticket, child_poison)
         : ticket.phase == TicketPhase::Empty
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

Status PersistController::finish(const std::span<Ticket> tickets,
                                 bool &child_poison) noexcept {
  Status result = Status::success();
  for (;;) {
    Ticket *selected = nullptr;
    for (Ticket &ticket : tickets) {
      if (ticket.phase == TicketPhase::OutputPersisting &&
          (selected == nullptr || ticket.batch < selected->batch)) {
        selected = &ticket;
      }
    }
    if (selected == nullptr) {
      return result;
    }
    const Status retired = retire(*selected, child_poison);
    if (result && !retired) {
      result = retired;
    }
    if (selected->phase == TicketPhase::OutputPersisting) {
      return result ? Status::fail(Reason::CompletionInvalid) : result;
    }
  }
}

bool PersistController::abort(const std::span<Ticket> tickets,
                              bool &child_poison) noexcept {
  (void)finish(tickets, child_poison);
  return pool_.graph_persist == nullptr ||
         std::all_of(
             pool_.graph_persist->slots.begin(),
             pool_.graph_persist->slots.end(),
             [](const residency::Persister &slot) { return slot.quiescent(); });
}

} // namespace rund::compute::detail::graph_reduce
