#include "../persist.hpp"

#include "../../projection.hpp"

#include "../../authority.hpp"
#include "../../../../../device/residency/registry/graph_persist_owner.hpp"

#include <algorithm>
#include <span>

namespace rund::compute::detail::graph_reduce::output_persist_detail {

Status issue(residency::Authority &authority,
             const std::shared_ptr<const residency::ResidencyPlan> &owner,
             const VirtualRunProjection &run,
             const residency::TiledGraphPlan &graph,
             const std::size_t terminal_stage, Ticket &ticket,
             residency::execution::GraphPersist &persist) noexcept {
  if (ticket.phase != TicketPhase::HostOutputDirty || !ticket.output_dirty ||
      ticket.count == 0u || terminal_stage >= graph.stages().size() ||
      graph.stages()[terminal_stage].ports.empty()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t resource =
      graph.stages()[terminal_stage].ports.back().resource;
  const bool cpu = ticket.collective != nullptr &&
                   ticket.collective->device != nullptr &&
                   ticket.collective->device->backend == Backend::Cpu;
  if (cpu && (ticket.book_domain == 0u ||
              !persist.bind_book_domain(ticket.book_domain))) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (cpu) {
    const residency::GraphPersistIdentity id =
        persist_identity(run, graph, terminal_stage, ticket, owner->identity());
    if (!persist.bind_identity(id)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  auto persist_owner = authority.graph_persists();
  const residency::AuthorityResult issued = persist_owner.issue_graph_persist(
      owner, run.active.graph, ticket.batch, terminal_stage, resource,
      std::span<const residency::PageUse>{ticket.collective_outputs.data(),
                                          ticket.count},
      run.graph_output, ticket.resident_output_region, persist);
  if (!issued) {
    return authority_status(issued);
  }
  return persist.plan() == owner->identity() &&
                 persist.coordinate() == ticket.collective_epoch.ordinal &&
                 persist.pages().size() == ticket.count
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

} // namespace rund::compute::detail::graph_reduce::output_persist_detail
