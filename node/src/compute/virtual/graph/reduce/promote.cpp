#include "promote.hpp"

#include "promote/internal.hpp"
#include "../../../device/residency/registry/graph_promote_owner.hpp"

#include <array>
#include <span>
#include <utility>

namespace rund::compute::detail::graph_reduce {

InputPromotionResult transfer_input_promotion(
    residency::Authority &authority, PipelineState &pipeline,
    const VirtualRunProjection &run, const residency::EpochLease destination,
    Stats &stats, Ticket &ticket) noexcept {
  promote_detail::Projection projection{};
  if (!promote_detail::project(run, destination, ticket, projection)) {
    return promote_detail::terminal(authority, ticket, projection,
                                    Status::fail(Reason::PipelineInvalid),
                                    Interval{});
  }
  Interval interval{};
  const Status copied = promote_detail::copy(pipeline, run, destination,
                                             projection, stats, interval);
  return promote_detail::terminal(authority, ticket, projection, copied,
                                  interval);
}

bool cancel_input_promotion(residency::Authority &authority,
                            Ticket &ticket) noexcept {
  if (!ticket.input_promote) {
    return true;
  }
  std::array<residency::execution::GraphPromoteCompletion,
             residency::execution::GraphPromoteCapacity>
      completions{};
  const auto pages = ticket.input_promote.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    completions[index] = residency::execution::GraphPromoteCompletion{
        .key = pages[index].key,
        .bytes = 0u,
        .source_frame = pages[index].source_frame,
        .target_frame = pages[index].target_frame,
    };
  }
  static_cast<void>(authority.graph_promotes().terminal_graph_promote(
      ticket.input_promote, Status::fail(Reason::CompletionInvalid),
      residency::execution::TerminalKind::UnknownMayWrite, true,
      std::span<const residency::execution::GraphPromoteCompletion>{
          completions.data(), pages.size()}));
  return authority.graph_promotes().release_graph_promote(
      std::move(ticket.input_promote));
}

} // namespace rund::compute::detail::graph_reduce
