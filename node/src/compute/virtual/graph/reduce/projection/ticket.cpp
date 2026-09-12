#include "../projection/internal.hpp"

#include "../../../state.hpp"

#include <algorithm>
#include <span>

namespace rund::compute::detail::graph_reduce {

Status project_ticket_impl(
    VirtualPipelineState &state, const VirtualRunProjection &run,
    const residency::TiledGraphPlan &graph, const residency::Pool &pool,
    const std::uint64_t batch, const std::size_t terminal_stage,
    const std::uint32_t capacity, Ticket &ticket) noexcept {
  if (ticket.phase != TicketPhase::Empty || ticket.host_ready_count != 0u ||
      ticket.input_promote || ticket.output_drain ||
      ticket.prefix_lease.token != 0u || ticket.collective_lease.token != 0u ||
      ticket.submitted != ExecutionStage::None) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (!reset_ticket(ticket)) {
    return Status::fail(Reason::PipelineBusy);
  }
  ticket.batch = batch;
  ticket.bank = static_cast<std::uint32_t>(batch % residency::Pool::BankCount);
  ticket.prefix = ticket.bank == 0u ? state.pipeline : state.alternate_pipeline;
  ticket.collective = graph_terminal_pipeline(state, ticket.bank);
  if (ticket.prefix == nullptr || ticket.collective == nullptr ||
      ticket.prefix->residency_bank != ticket.bank ||
      ticket.collective->residency_bank != ticket.bank ||
      ticket.prefix->residency_graph_stage != 0u ||
      ticket.collective->residency_graph_stage != terminal_stage ||
      !run.active.graph.batch(batch, ticket.pages) ||
      ticket.pages.page_count == 0u ||
      ticket.pages.page_count > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.count = static_cast<std::size_t>(ticket.pages.page_count);
  StageScratch terminal_scratch{.uses = ticket.stage_uses,
                               .requests = ticket.stage_requests};
  if (!project_stage_scratch(graph, run, pool, ticket, terminal_stage, capacity,
                             terminal_scratch)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const auto terminal_write =
      std::find_if(graph.stages()[terminal_stage].ports.begin(),
                   graph.stages()[terminal_stage].ports.end(),
                   [](const residency::TiledGraphPort port) {
                     return port.access == residency::Access::Write;
                   });
  if (terminal_write == graph.stages()[terminal_stage].ports.end()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t terminal_write_port = static_cast<std::size_t>(
      terminal_write - graph.stages()[terminal_stage].ports.begin());
  ticket.collective_epoch = terminal_scratch.epoch;
  ticket.collective_use_count = terminal_scratch.use_count;
  ticket.collective_request_count = terminal_scratch.port_count;
  ticket.collective_anchor_port = terminal_scratch.anchor_port;
  ticket.collective_output_port = terminal_write_port;
  std::copy(terminal_scratch.uses.begin() +
                static_cast<std::ptrdiff_t>(terminal_write_port * ticket.count),
            terminal_scratch.uses.begin() +
                static_cast<std::ptrdiff_t>((terminal_write_port + 1u) *
                                            ticket.count),
            ticket.collective_outputs.begin());
  if (!copy_keys(
          std::span<const residency::PageUse>{ticket.collective_outputs.data(),
                                              ticket.count},
          run.graph_output,
          std::span<residency::CacheKey>{ticket.output_keys.data(),
                                         ticket.count}) ||
      !project_virtual_epoch(run, batch, ticket.byte_epoch) ||
      ticket.byte_epoch.failed_page != ticket.pages.first_page ||
      ticket.byte_epoch.page_count != ticket.pages.page_count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  StageScratch prefix_scratch{.uses = ticket.stage_uses,
                               .requests = ticket.stage_requests};
  if (!project_stage_scratch(graph, run, pool, ticket, 0u, capacity,
                             prefix_scratch) ||
      prefix_scratch.anchor_port != 0u ||
      graph.stages().front().ports.front().access != residency::Access::Read) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const auto prefix_write = std::find_if(
      graph.stages().front().ports.begin(), graph.stages().front().ports.end(),
      [](const residency::TiledGraphPort port) {
        return port.access == residency::Access::Write;
      });
  if (prefix_write == graph.stages().front().ports.end()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.prefix_epoch = prefix_scratch.epoch;
  ticket.prefix_use_count = prefix_scratch.use_count;
  ticket.prefix_request_count = prefix_scratch.port_count;
  ticket.prefix_anchor_port = prefix_scratch.anchor_port;
  ticket.input_region = resource_region(
      pool, graph, graph.stages().front().ports[0].resource, ticket.bank);
  ticket.intermediate_region =
      resource_region(pool, graph, prefix_write->resource, ticket.bank);
  ticket.output_region =
      resource_region(pool, graph, terminal_write->resource, ticket.bank);
  if (ticket.input_region.count != capacity ||
      ticket.intermediate_region.count != capacity ||
      ticket.output_region.count != capacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.resident_output_region = ticket.output_region;
  ticket.phase = TicketPhase::Projected;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
