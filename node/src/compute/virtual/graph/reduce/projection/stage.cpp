#include "../projection/internal.hpp"

namespace rund::compute::detail::graph_reduce {

bool project_stage_scratch(const residency::TiledGraphPlan &graph,
                           const VirtualRunProjection &run,
                           const residency::Pool &pool, const Ticket &ticket,
                           const std::size_t stage_index,
                           const std::uint32_t capacity,
                           StageScratch &scratch) noexcept {
  scratch = {};
  if (stage_index >= graph.stages().size() || ticket.count == 0u ||
      ticket.count > PipelineLeafCapacity) {
    return false;
  }
  const residency::TiledGraphStage &stage = graph.stages()[stage_index];
  if (stage.ports.size() < 2u ||
      stage.ports.size() > residency::TiledGraphPortCapacity ||
      stage.ports.size() > scratch.uses.size() / ticket.count) {
    return false;
  }
  scratch.port_count = stage.ports.size();
  scratch.use_count = scratch.port_count * ticket.count;
  if (!run.active.graph.project(
          ticket.batch, stage_index,
          std::span<residency::PageUse>{scratch.uses.data(), scratch.use_count},
          scratch.epoch)) {
    return false;
  }
  bool anchored = false;
  for (std::size_t port_index = 0u; port_index < scratch.port_count;
       ++port_index) {
    const residency::TiledGraphPort port = stage.ports[port_index];
    const residency::GraphMaterialization *const materialization =
        graph_materialization(run, port.resource);
    const residency::FrameRegion region =
        resource_region(pool, graph, port.resource, ticket.bank);
    const residency::TiledGraphResource *const resource =
        graph.resource(port.resource);
    std::array<residency::FrameRegion, residency::Pool::BankCount>
        cache_regions{};
    std::size_t cache_region_count = 0u;
    if (materialization == nullptr || resource == nullptr ||
        region.count != capacity ||
        !resource_cache_regions(pool, graph, port.resource, cache_regions,
                                cache_region_count)) {
      return false;
    }
    scratch.requests[port_index] = residency::GraphPortRequest{
        .program_port = port.program_port,
        .first_use = port_index * ticket.count,
        .use_count = ticket.count,
        .materialization = *materialization,
        .region = region,
        .cache_regions = cache_regions,
        .cache_region_count = cache_region_count,
        .remaps = resource->remaps,
    };
    if (!anchored && port.access == residency::Access::Read) {
      scratch.anchor_port = port_index;
      anchored = true;
    }
  }
  return anchored;
}

} // namespace rund::compute::detail::graph_reduce
