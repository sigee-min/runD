#include "../output.hpp"

#include "../authority.hpp"

#include "../../../../device/residency/registry/graph_drain_owner.hpp"

#include <limits>
#include <span>

namespace rund::compute::detail::graph_reduce {

Status issue_output_drain(VirtualPipelineState &state, residency::Pool &pool,
                          const VirtualRunProjection &run,
                          const residency::TiledGraphPlan &graph,
                          const std::size_t stage, Ticket &ticket) noexcept {
  if (ticket.phase != TicketPhase::DeviceOutputDirty || !ticket.output_dirty ||
      ticket.collective == nullptr || state.pipeline == nullptr ||
      state.pipeline->residency == nullptr ||
      ticket.collective->device == nullptr ||
      ticket.collective->device->backend == Backend::Cpu ||
      ticket.collective->residency_output >=
          ticket.collective->resources.size() ||
      graph.stages().empty() || stage >= graph.stages().size() ||
      graph.stages()[stage].ports.empty() || run.frame_capacity == 0u ||
      run.frame_capacity > std::numeric_limits<std::uint32_t>::max()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::FrameRegion host_region{
      .tier = residency::FrameTier::Host,
      .role = residency::FrameRole::Output,
      .first = pool.first_host_output_frame +
               ticket.bank * static_cast<std::uint32_t>(run.frame_capacity),
      .count = static_cast<std::uint32_t>(run.frame_capacity),
  };
  const std::uint32_t resource = graph.stages()[stage].ports.back().resource;
  const residency::AuthorityResult drain =
      pool.authority().graph_drains().issue_graph_drain(
          state.pipeline->residency, run.active.graph, ticket.batch, stage,
          resource,
          std::span<const residency::PageUse>{ticket.collective_outputs.data(),
                                              ticket.count},
          run.graph_output, ticket.output_region, host_region,
          ticket.output_drain);
  if (!drain) {
    return authority_status(drain);
  }
  const auto pages = ticket.output_drain.pages();
  if (pages.size() != ticket.count ||
      ticket.output_drain.plan() != state.pipeline->residency->identity() ||
      ticket.output_drain.coordinate() != ticket.collective_epoch.ordinal) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    ticket.resident_outputs[index] = residency::CacheBinding{
        .key = pages[index].key,
        .frame = pages[index].target_frame,
        .access = residency::Access::Write,
        .dirty = {.offset = ticket.collective_outputs[index].dirty.offset,
                  .bytes = ticket.collective_outputs[index].dirty.bytes},
    };
  }
  ticket.resident_output_region = host_region;
  ticket.phase = TicketPhase::OutputDraining;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
