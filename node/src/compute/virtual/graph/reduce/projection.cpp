#include "projection.hpp"

#include "projection/internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status project_ticket(VirtualPipelineState &state,
                      const VirtualRunProjection &run,
                      const residency::TiledGraphPlan &graph,
                      const residency::Pool &pool, const std::uint64_t batch,
                      const std::size_t terminal_stage,
                      const std::uint32_t capacity, Ticket &ticket) noexcept {
  return project_ticket_impl(state, run, graph, pool, batch, terminal_stage,
                             capacity, ticket);
}

} // namespace rund::compute::detail::graph_reduce
