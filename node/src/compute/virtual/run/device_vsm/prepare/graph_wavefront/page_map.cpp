#include "../graph_wavefront.hpp"

#include "../../../../../../accel/kernel/residency/device_vsm/graph_wavefront.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace rund::compute::detail::device_vsm_product_detail {

bool project_graph_page_map(
    const residency::TiledGraphPlan &plan,
    const std::span<const std::uint32_t> graph_input_resources,
    node::accel::detail::DeviceVsmPageMap &map) noexcept {
  namespace accel = node::accel::detail;
  map = {};
  std::uint32_t rows = 0u;
  std::uint32_t total = 0u;
  for (const residency::TiledGraphResource &resource : plan.resources()) {
    if (resource.remaps.empty()) {
      continue;
    }
    if (resource.kind != residency::GraphResourceKind::ExternalInput ||
        resource.persistence != residency::ResourcePersistence::Backing ||
        resource.remaps.size() > accel::DeviceVsmPageMapEntryCapacity ||
        rows >= accel::DeviceVsmPageMapRowCapacity ||
        total > std::numeric_limits<std::uint32_t>::max() -
                    resource.remaps.size()) {
      return false;
    }
    const auto found =
        std::find(graph_input_resources.begin(), graph_input_resources.end(),
                  resource.resource);
    if (found == graph_input_resources.end() || plan.frame_capacity() == 0u ||
        plan.frame_capacity() > accel::DeviceVsmPageMapEntryCapacity) {
      return false;
    }
    accel::DeviceVsmPageMapRow row{};
    row.resource = resource.resource;
    row.external_slot = static_cast<std::uint32_t>(
        std::distance(graph_input_resources.begin(), found));
    row.frame_capacity = static_cast<std::uint32_t>(plan.frame_capacity());
    row.page_bytes = resource.page_bytes;
    row.count = static_cast<std::uint32_t>(resource.remaps.size());
    for (std::size_t index = 0u; index < resource.remaps.size(); ++index) {
      const residency::GraphPageRemap remap = resource.remaps[index];
      row.entries[index] = accel::DeviceVsmPageMapEntry{
          .target = remap.target_local,
          .source = remap.source_local,
          .origin = static_cast<std::uint32_t>(remap.source_origin)};
    }
    if (!accel::device_vsm_page_map_store_row(map, rows, row)) {
      return false;
    }
    ++rows;
    total += row.count;
  }
  accel::device_vsm_page_map_seal(map, rows != 0u, rows, total);
  return accel::device_vsm_page_map_valid(map, plan.page_count());
}

} // namespace rund::compute::detail::device_vsm_product_detail
