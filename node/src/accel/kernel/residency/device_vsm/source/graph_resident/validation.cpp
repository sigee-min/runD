#include "internal.hpp"

#include "../typed_map/scalar.hpp"

#include <limits>

namespace rund::node::accel::detail {

bool graph_resident_endpoints_valid(
    const DeviceVsmPageGeometry &geometry,
    const DeviceVsmResidentSet &residents) noexcept {
  if ((geometry.element_bytes != sizeof(std::uint32_t) &&
       geometry.element_bytes != sizeof(std::uint64_t)) ||
      geometry.logical_bytes == 0u ||
      geometry.logical_bytes % geometry.element_bytes != 0u ||
      residents.input_count == 0u || residents.output_count != 1u ||
      residents.count != residents.input_count + 1u ||
      residents.count > DeviceVsmResidentCapacity) {
    return false;
  }
  const std::uint64_t elements =
      geometry.logical_bytes / geometry.element_bytes;
  for (std::size_t index = 0u; index < residents.count; ++index) {
    const auto &row = residents.rows[index];
    const bool input = index < residents.input_count;
    if (row.role != (input ? DeviceVsmResidentRole::Input
                           : DeviceVsmResidentRole::Output) ||
        row.handle == nullptr || row.backing.id == 0u ||
        row.backing.bytes < geometry.logical_bytes ||
        row.backing.offset_bytes != 0u ||
        row.backing.element_bytes != geometry.element_bytes ||
        row.backing.stride_bytes != geometry.element_bytes ||
        row.backing.count < elements) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (row.backing.id == residents.rows[prior].backing.id ||
          device_vsm_graph_resident_same_object(
              row.handle, residents.rows[prior].handle)) {
        return false;
      }
    }
  }
  return true;
}

bool graph_resident_stages_valid(
    const std::span<const DeviceVsmGraphResidentStageSource> stages,
    const DeviceVsmGraphResidentProof &proof, const char *&reason) noexcept {
  reason = "device_vsm_graph_resident_stage_invalid";
  if (stages.size() < 2u || stages.size() != proof.stage_count ||
      stages.front().artifact == nullptr ||
      !device_vsm_graph_resident_type_valid(proof.type)) {
    return false;
  }
  const rund::kernel::ComputeApi api = stages.front().artifact->key.api;
  for (const auto &stage : stages) {
    if (stage.artifact == nullptr || stage.input == nullptr ||
        stage.semantic == nullptr ||
        !device_vsm_graph_pointwise::validate_stage(
            *stage.artifact, *stage.input, *stage.semantic) ||
        (stage.artifact->key.api != rund::kernel::ComputeApi::Vulkan &&
         stage.artifact->key.api != rund::kernel::ComputeApi::Metal) ||
        stage.artifact->key.scalar != proof.type.scalar ||
        stage.artifact->key.domain != proof.type.domain ||
        !rund::kernel::ComputeFixedFormatAbsent(
            stage.artifact->key.fixed_format)) {
      return false;
    }
    if (stage.artifact->key.api != api) {
      reason = "device_vsm_graph_resident_stage_api_mismatch";
      return false;
    }
    for (const auto &node : stage.input->parsed.nodes) {
      if (!device_vsm_typed_map::parameter_free_total_scalar_op_supported(
              static_cast<rund::kernel::IrOp>(node.op))) {
        return false;
      }
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
