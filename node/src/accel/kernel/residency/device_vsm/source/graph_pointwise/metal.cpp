#include "metal/internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_pointwise {

std::string
metal_source(const rund::kernel::ArtifactKey &key,
             const std::span<const DeviceVsmGraphPointwiseStage> stages,
             const DeviceVsmGraphPointwiseTopology &topology,
             const DeviceVsmPageGeometry &geometry,
             const DeviceVsmPageMap &page_map,
             const DeviceVsmGraphWavefrontProof &wavefront) {
  namespace metal_detail =
      ::rund::node::accel::detail::device_vsm_graph_pointwise::metal;
  std::string source;
  if (stages.size() < 2u || stages.front().input == nullptr ||
      topology.stage_count != stages.size() ||
      topology.external_input_count == 0u ||
      topology.external_input_count >= DeviceVsmResidentCapacity) {
    return {};
  }
  const std::uint32_t expected_element_bytes =
      key.scalar == rund::kernel::ComputeScalar::Lane32 ? sizeof(std::uint32_t)
      : key.scalar == rund::kernel::ComputeScalar::Lane64
          ? sizeof(std::uint64_t)
          : 0u;
  if (!((key.scalar == rund::kernel::ComputeScalar::Lane32 &&
         key.domain == rund::kernel::ComputeDomain::U32) ||
        (key.scalar == rund::kernel::ComputeScalar::Lane64 &&
         key.domain == rund::kernel::ComputeDomain::U64)) ||
      geometry.element_bytes != expected_element_bytes) {
    return {};
  }
  const auto io =
      build_io_ir(topology.external_input_count, key.scalar, key.domain,
                  static_cast<std::uint32_t>(geometry.element_bytes));
  const bool map_active = device_vsm_page_map_active(page_map);
  std::vector<rund::kernel::compute_lowering_detail::BindingLayout> layouts;
  if (!metal_detail::append_fixed_helpers(source, stages, key) ||
      !device_vsm_typed_map::append_metal_header(source, key, io, layouts,
                                                 !map_active) ||
      !metal_detail::append_map_binding(source, map_active) ||
      !metal_detail::append_ring_bindings(source,
                                          topology.external_input_count + 1u,
                                          topology.external_input_count) ||
      !metal_detail::append_page_ring(source, key, stages, topology, page_map,
                                      wavefront, io, layouts, map_active)) {
    return {};
  }
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise
