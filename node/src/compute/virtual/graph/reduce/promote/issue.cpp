#include "../promote.hpp"
#include "../../../../device/residency/registry/graph_promote_owner.hpp"

#include <utility>

namespace rund::compute::detail::graph_reduce {

Status issue_input_promotion(residency::Authority &authority,
                             const residency::TiledGraphInvocation &invocation,
                             Ticket &ticket, const std::size_t stage,
                             const std::uint64_t destination_token) noexcept {
  if (ticket.host_ready_count == 0u ||
      ticket.host_ready_count > ticket.host_ready.size() ||
      ticket.input_promote || ticket.forecast_stage != stage ||
      destination_token == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!authority.graph_promotes().issue_graph_promote_group(
          std::span<residency::execution::GraphReady>{ticket.host_ready.data(),
                                                      ticket.host_ready_count},
          invocation, ticket.batch, stage, destination_token,
          ticket.input_promote)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.host_ready_count = 0u;
  ticket.forecast_stage = 0u;
  ticket.forecast_resources = {};
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
