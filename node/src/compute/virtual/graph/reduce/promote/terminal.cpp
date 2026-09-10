#include "internal.hpp"
#include "../../../../device/residency/registry/graph_promote_owner.hpp"

#include <span>
#include <utility>

namespace rund::compute::detail::graph_reduce::promote_detail {

InputPromotionResult terminal(residency::Authority &authority, Ticket &ticket,
                              Projection &projection, const Status status,
                              const Interval interval) noexcept {
  const auto pages = ticket.input_promote.pages();
  if (pages.size() > projection.completions.size()) {
    return {};
  }
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    projection.completions[index] =
        residency::execution::GraphPromoteCompletion{
            .key = pages[index].key,
            .bytes = status ? pages[index].bytes : 0u,
            .source_frame = pages[index].source_frame,
            .target_frame = pages[index].target_frame,
        };
  }
  const bool accepted = authority.graph_promotes().terminal_graph_promote(
      ticket.input_promote, status,
      status ? residency::execution::TerminalKind::Known
             : residency::execution::TerminalKind::UnknownMayWrite,
      !status,
      std::span<const residency::execution::GraphPromoteCompletion>{
          projection.completions.data(), pages.size()});
  return InputPromotionResult{
      .status = status,
      .interval = interval,
      .transfer_complete = static_cast<bool>(status),
      .released = accepted && authority.graph_promotes().release_graph_promote(
                                      std::move(ticket.input_promote)),
  };
}

} // namespace rund::compute::detail::graph_reduce::promote_detail
