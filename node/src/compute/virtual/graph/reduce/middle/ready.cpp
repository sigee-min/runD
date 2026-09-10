#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

bool MiddleController::ready_external_inputs(
    Ticket &ticket, const std::size_t stage_index) noexcept {
  StageScratch scratch{};
  if (!project_stage_scratch(graph_, run_, pool_, ticket, stage_index,
                             capacity_, scratch)) {
    return false;
  }
  bool external = false;
  for (std::size_t port_index = 0u; port_index < scratch.port_count;
       ++port_index) {
    const residency::TiledGraphPort port =
        graph_.stages()[stage_index].ports[port_index];
    const residency::TiledGraphResource *const resource =
        graph_.resource(port.resource);
    if (!residency::reads(port.access) || resource == nullptr ||
        resource->kind != residency::GraphResourceKind::ExternalInput ||
        resource->persistence != residency::ResourcePersistence::Backing) {
      continue;
    }
    external = true;
    const residency::GraphPortRequest &request = scratch.requests[port_index];
    std::array<residency::CacheKey, PipelineLeafCapacity> keys{};
    std::array<std::uint8_t, PipelineLeafCapacity> resident{};
    std::array<std::uint8_t, PipelineLeafCapacity> region_resident{};
    for (std::size_t page = 0u; page < ticket.count; ++page) {
      if (!residency::project_graph_cache_key(
              request.materialization,
              scratch.uses[port_index * ticket.count + page].key, keys[page])) {
        return false;
      }
    }
    for (std::size_t region = 0u; region < request.cache_region_count;
         ++region) {
      region_resident.fill(0u);
      if (!authority_.probe(
              std::span<const residency::CacheKey>{keys.data(), ticket.count},
              std::span<std::uint8_t>{region_resident.data(), ticket.count},
              request.cache_regions[region].first,
              request.cache_regions[region].count)) {
        return false;
      }
      for (std::size_t page = 0u; page < ticket.count; ++page) {
        resident[page] = static_cast<std::uint8_t>(resident[page] != 0u ||
                                                   region_resident[page] != 0u);
      }
    }
    if (std::any_of(resident.begin(),
                    resident.begin() +
                        static_cast<std::ptrdiff_t>(ticket.count),
                    [](const std::uint8_t value) { return value == 0u; })) {
      return false;
    }
  }
  return external ? wavefront_.device_resident(
                        ticket.batch, static_cast<std::uint32_t>(stage_index))
                  : wavefront_.dependency_ready(
                        ticket.batch, static_cast<std::uint32_t>(stage_index));
}

} // namespace rund::compute::detail::graph_reduce
