#include "internal.hpp"

#include "../../../../device/residency/registry/graph_drain_owner.hpp"

#include <span>
#include <utility>

namespace rund::compute::detail::graph_reduce::output_detail {

OutputDrainResult terminal(residency::Authority &authority, Ticket &ticket,
                           Projection &projection,
                           const Download download) noexcept {
  auto drains = authority.graph_drains();
  const bool accepted = drains.terminal_graph_drain(
      ticket.output_drain, download.status,
      residency::execution::TerminalKind::Known, !download.complete,
      std::span<const residency::execution::GraphDrainCompletion>{
          projection.completions.data(), projection.page_count});
  return OutputDrainResult{
      .status = download.status,
      .interval = download.interval,
      .transfer_complete = download.complete,
      .released = accepted &&
                  drains.release_graph_drain(std::move(ticket.output_drain)),
  };
}

} // namespace rund::compute::detail::graph_reduce::output_detail
