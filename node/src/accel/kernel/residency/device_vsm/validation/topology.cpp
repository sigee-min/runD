#include "proof.hpp"

namespace rund::node::accel::detail {

bool device_vsm_topology_valid(const DeviceVsmProof &proof) noexcept {
  if (proof.topology == DeviceVsmTopology::Pointwise) {
    return !proof.window.semantic.ok &&
           proof.graph_map_reduce.workgroup_width == 0u &&
           proof.graph_pointwise.workgroup_width == 0u &&
           proof.scan.workgroup_width == 0u &&
           proof.reduce.workgroup_width == 0u &&
           proof.output_bytes == proof.geometry.logical_bytes &&
           device_vsm_complete_frame_geometry(proof.geometry);
  }
  if (proof.topology == DeviceVsmTopology::GraphMapReduce) {
    const rund::kernel::ReducePlan &semantic = proof.graph_map_reduce.semantic;
    const bool operation =
        semantic.op == rund::kernel::ReduceOp::Sum ||
        semantic.op == rund::kernel::ReduceOp::CountNonzero ||
        semantic.op == rund::kernel::ReduceOp::Min ||
        semantic.op == rund::kernel::ReduceOp::Max;
    return !proof.window.semantic.ok &&
           proof.graph_pointwise.workgroup_width == 0u &&
           device_vsm_complete_frame_geometry(proof.geometry) && semantic.ok &&
           operation && semantic.element == rund::kernel::ReduceElement::U64 &&
           semantic.element_bytes == sizeof(std::uint64_t) &&
           semantic.element_count ==
               proof.geometry.frame_bytes / sizeof(std::uint64_t) &&
           semantic.count_source == rund::kernel::ComputeCountSource::BufferU64 &&
           proof.graph_map_reduce.workgroup_width == 256u &&
           proof.scan.workgroup_width == 0u &&
           proof.reduce.workgroup_width == 0u &&
           device_vsm_graph_wavefront_valid(proof.graph_map_reduce.wavefront,
                                            proof.geometry.page_count) &&
           proof.output_bytes == sizeof(std::uint64_t) &&
           proof.plan.input_buffer_count != 0u &&
           proof.plan.input_buffer_count == proof.residents.input_count &&
           proof.plan.output_buffer_count == 1u;
  }
  if (proof.topology == DeviceVsmTopology::GraphPointwise) {
    const auto map_valid = [&]() noexcept {
      const DeviceVsmPageMap &map = proof.graph_pointwise.page_map;
      if (!device_vsm_page_map_valid(map, proof.geometry.page_count)) {
        return false;
      }
      if (!device_vsm_page_map_active(map)) {
        return true;
      }
      if (proof.parameter_bytes != 0u || proof.plan.param_bytes != 0u ||
          map.words[3u] == 0u ||
          map.words[3u] > proof.graph_pointwise.topology.external_input_count) {
        return false;
      }
      for (std::size_t index = 0u; index < map.words[3u]; ++index) {
        DeviceVsmPageMapRow row{};
        if (!device_vsm_page_map_load_row(map, index, row) ||
            row.external_slot >=
                proof.graph_pointwise.topology.external_input_count ||
            row.frame_capacity !=
                proof.graph_pointwise.wavefront.frame_capacity ||
            row.page_bytes != proof.geometry.frame_bytes || row.resource == 0u) {
          return false;
        }
      }
      return true;
    };
    return !proof.window.semantic.ok &&
           proof.graph_map_reduce.workgroup_width == 0u &&
           proof.graph_pointwise.workgroup_width == 256u &&
           proof.graph_pointwise.stage_count >= 2u &&
           proof.graph_pointwise.stage_count <= DeviceVsmGraphStageCapacity &&
           proof.scan.workgroup_width == 0u &&
           proof.reduce.workgroup_width == 0u &&
           device_vsm_complete_frame_geometry(proof.geometry) &&
           device_vsm_graph_wavefront_valid(proof.graph_pointwise.wavefront,
                                            proof.geometry.page_count) &&
           device_vsm_graph_pointwise_topology_valid(
               proof.graph_pointwise.topology,
               proof.graph_pointwise.wavefront) &&
           proof.graph_pointwise.wavefront.stage_count ==
               proof.graph_pointwise.stage_count &&
           proof.graph_pointwise.topology.stage_count ==
               proof.graph_pointwise.stage_count &&
           proof.output_bytes == proof.geometry.logical_bytes && map_valid() &&
           proof.parameter_bytes == 0u && proof.plan.param_bytes == 0u &&
           proof.plan.input_buffer_count ==
               proof.graph_pointwise.topology.external_input_count &&
           proof.plan.output_buffer_count == 1u &&
           proof.plan.dispatch_count == 1u;
  }
  if (proof.topology == DeviceVsmTopology::GraphResident) {
    return !proof.window.semantic.ok &&
           proof.graph_map_reduce.workgroup_width == 0u &&
           proof.graph_pointwise.workgroup_width == 0u &&
           proof.scan.workgroup_width == 0u &&
           proof.reduce.workgroup_width == 0u &&
           device_vsm_complete_frame_geometry(proof.geometry) &&
           device_vsm_graph_wavefront_valid(proof.graph_wavefront,
                                            proof.geometry.page_count) &&
           device_vsm_graph_resident_type_valid(proof.graph_resident.type) &&
           proof.plan.scalar == proof.graph_resident.type.scalar &&
           proof.plan.domain == proof.graph_resident.type.domain &&
           proof.geometry.element_bytes ==
               proof.graph_resident.type.element_bytes &&
           proof.geometry.payload_bytes % proof.geometry.element_bytes == 0u &&
           device_vsm_graph_resident_valid(
               proof.graph_resident,
               proof.geometry.payload_bytes / proof.geometry.element_bytes,
               proof.residents.input_count, proof.geometry.page_count) &&
           (!device_vsm_page_map_active(proof.graph_resident.page_map) ||
            (proof.parameter_bytes == 0u && proof.plan.param_bytes == 0u)) &&
           (proof.plan.api == rund::kernel::ComputeApi::Metal ||
            proof.plan.api == rund::kernel::ComputeApi::Vulkan) &&
           proof.output_bytes == proof.geometry.logical_bytes &&
           proof.plan.input_buffer_count == proof.residents.input_count &&
           proof.plan.output_buffer_count == 1u &&
           proof.plan.dispatch_count == 1u;
  }
  if (proof.topology == DeviceVsmTopology::Scan) {
    return !proof.window.semantic.ok &&
           proof.graph_map_reduce.workgroup_width == 0u &&
           proof.graph_pointwise.workgroup_width == 0u &&
           proof.reduce.workgroup_width == 0u &&
           device_vsm_scan_proof_valid(proof);
  }
  if (proof.topology == DeviceVsmTopology::Reduce) {
    return !proof.window.semantic.ok &&
           proof.graph_map_reduce.workgroup_width == 0u &&
           proof.graph_pointwise.workgroup_width == 0u &&
           proof.scan.workgroup_width == 0u &&
           device_vsm_reduce_proof_valid(proof);
  }
  return proof.output_bytes == proof.geometry.logical_bytes &&
         proof.graph_map_reduce.workgroup_width == 0u &&
         proof.graph_pointwise.workgroup_width == 0u &&
         proof.scan.workgroup_width == 0u &&
         proof.reduce.workgroup_width == 0u &&
         device_vsm_window_proof_valid(proof);
}

} // namespace rund::node::accel::detail
