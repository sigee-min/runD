#include "active.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::compute::detail {

bool project_virtual_active(const residency::StreamPlan &capacity,
                            const std::uint64_t active_count,
                            const std::uint64_t capacity_count,
                            const std::uint64_t input_payload_elements,
                            const std::uint64_t active_output_count,
                            const std::uint64_t input_element_bytes,
                            const std::uint64_t output_element_bytes,
                            const bool transient_page_outputs,
                            VirtualActiveProjection &projection) noexcept {
  projection = {};
  if (active_count > capacity_count || capacity.frame_capacity() == 0u ||
      input_element_bytes == 0u || output_element_bytes == 0u ||
      input_payload_elements == 0u) {
    return false;
  }
  const std::uint64_t full_pages =
      capacity_count / input_payload_elements +
      static_cast<std::uint64_t>(capacity_count % input_payload_elements != 0u);
  const std::uint64_t active_pages =
      active_count / input_payload_elements +
      static_cast<std::uint64_t>(active_count % input_payload_elements != 0u);
  if (capacity.page_count() != full_pages ||
      !kernel::checked::mul(active_count, input_element_bytes,
                            projection.input_bytes) ||
      !kernel::checked::mul(active_output_count, output_element_bytes,
                            projection.output_bytes)) {
    projection = {};
    return false;
  }
  std::uint64_t dirty_bytes = projection.output_bytes;
  if (transient_page_outputs &&
      !kernel::checked::mul(active_pages, capacity.dirty_extent().bytes,
                            dirty_bytes)) {
    projection = {};
    return false;
  }
  projection.stream = residency::StreamPlan{
      active_pages, capacity.frame_capacity(), capacity.dirty_extent(),
      dirty_bytes, capacity.prefetch_distance()};
  projection.active_count = active_count;
  projection.resident_frames_peak =
      std::min(active_pages, capacity.frame_capacity());
  return true;
}

bool project_virtual_graph_active(
    const residency::TiledGraphPlan &capacity, const std::uint64_t active_count,
    const std::uint64_t capacity_count,
    const std::uint64_t input_payload_elements,
    const std::uint64_t input_element_bytes,
    const std::uint64_t output_element_bytes,
    VirtualActiveProjection &projection) noexcept {
  projection = {};
  if (active_count > capacity_count || capacity.frame_capacity() == 0u ||
      input_element_bytes == 0u || output_element_bytes == 0u ||
      input_payload_elements == 0u) {
    return false;
  }
  const std::uint64_t full_pages =
      capacity_count / input_payload_elements +
      static_cast<std::uint64_t>(capacity_count % input_payload_elements != 0u);
  const std::uint64_t active_pages =
      active_count / input_payload_elements +
      static_cast<std::uint64_t>(active_count % input_payload_elements != 0u);
  if (capacity.page_count() != full_pages || capacity.resources().empty() ||
      capacity.resources().size() > residency::TiledGraphResourceCapacity ||
      capacity.stages().empty() ||
      !kernel::checked::mul(active_count, input_element_bytes,
                            projection.input_bytes)) {
    projection = {};
    return false;
  }
  std::array<std::uint64_t, residency::TiledGraphResourceCapacity> extents{};
  const residency::TiledGraphStage &last_stage = capacity.stages().back();
  const auto output_port =
      std::find_if(last_stage.ports.begin(), last_stage.ports.end(),
                   [](const residency::TiledGraphPort &port) {
                     return residency::writes(port.access);
                   });
  if (output_port == last_stage.ports.end()) {
    projection = {};
    return false;
  }
  const std::uint32_t output_resource = output_port->resource;
  for (std::size_t index = 0u; index < capacity.resources().size(); ++index) {
    const residency::TiledGraphResource resource = capacity.resources()[index];
    if (resource.resource == output_resource) {
      const bool projected =
          last_stage.domain == residency::StageDomain::TilePartial
              ? kernel::checked::mul(active_pages, resource.page_bytes,
                                     extents[index])
              : kernel::checked::mul(active_count, output_element_bytes,
                                     extents[index]);
      if (!projected) {
        projection = {};
        return false;
      }
    } else if (resource.kind == residency::GraphResourceKind::ExternalInput ||
               resource.persistence ==
                   residency::ResourcePersistence::Transient) {
      extents[index] = projection.input_bytes;
    } else {
      projection = {};
      return false;
    }
  }
  if (!capacity.active(active_pages,
                       std::span<const std::uint64_t>{
                           extents.data(), capacity.resources().size()},
                       projection.graph) ||
      !projection.graph.supply_stream(projection.stream)) {
    projection = {};
    return false;
  }
  projection.active_count = active_count;
  if (last_stage.domain == residency::StageDomain::TilePartial) {
    projection.output_bytes = output_element_bytes;
  } else if (!kernel::checked::mul(active_count, output_element_bytes,
                                   projection.output_bytes)) {
    projection = {};
    return false;
  }
  projection.resident_frames_peak =
      std::min(active_pages, capacity.frame_capacity());
  return true;
}

} // namespace rund::compute::detail
