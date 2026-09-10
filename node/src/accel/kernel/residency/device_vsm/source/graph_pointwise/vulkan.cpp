#include "vulkan/internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_pointwise {

std::string
vulkan_source(const rund::kernel::ArtifactKey &key,
              const std::span<const DeviceVsmGraphPointwiseStage> stages,
              const DeviceVsmGraphPointwiseTopology &topology,
              const DeviceVsmPageGeometry &geometry,
              const DeviceVsmPageMap &page_map,
              const DeviceVsmGraphWavefrontProof &wavefront) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  namespace vulkan_detail = device_vsm_graph_pointwise::vulkan;
  std::string source;
  std::vector<rund::kernel::compute_lowering_detail::BindingLayout> layouts;
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
  if (!device_vsm_typed_map::append_vulkan_prelude(source, key, io, layouts,
                                                   !map_active) ||
      !vulkan_detail::append_stage_fixed_helpers(source, stages, key) ||
      !vulkan_detail::install_map_binding(source, map_active)) {
    return {};
  }
  const vulkan_detail::MapRows map_rows =
      vulkan_detail::collect_map_rows(page_map);
  const std::size_t data_buffers = topology.external_input_count + 1u;
  if (!vulkan_detail::append_storage_bindings(source, data_buffers,
                                              topology.external_input_count)) {
    return {};
  }
  std::vector<std::string> external_values(topology.external_input_count);
  for (std::size_t input = 0u; input < external_values.size(); ++input) {
    external_values[input] =
        map_active ? device_vsm_typed_map::vulkan_scratch_value(
                         key.scalar, "input_scratch_" + std::to_string(input))
                   : lowering::VulkanReadNodeExpr(key, layouts[input],
                                                  io.bindings[input]);
  }
  if (!vulkan_detail::append_body(source, key, stages, topology, page_map,
                                  wavefront, io, layouts, external_values,
                                  map_rows)) {
    return {};
  }
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise
