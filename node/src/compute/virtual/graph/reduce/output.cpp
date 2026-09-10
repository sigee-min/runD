#include "output.hpp"

#include "output/internal.hpp"

namespace rund::compute::detail::graph_reduce {

OutputDrainResult transfer_output_drain(residency::Authority &authority,
                                        const VirtualRunProjection &run,
                                        Ticket &ticket) noexcept {
  OutputDrainResult invalid{};
  if (ticket.phase != TicketPhase::OutputDraining ||
      ticket.collective == nullptr) {
    return invalid;
  }
  output_detail::Projection projection{};
  if (!output_detail::project(run, ticket, projection)) {
    return output_detail::terminal(
        authority, ticket, projection,
        output_detail::Download{.status =
                                    Status::fail(Reason::PipelineInvalid)});
  }
  output_detail::Download download =
      output_detail::execute(*ticket.collective, projection);
  return output_detail::terminal(authority, ticket, projection, download);
}

} // namespace rund::compute::detail::graph_reduce
