#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_graph_pointwise {

bool validate_stage(
    const rund::kernel::LoweringArtifact &source,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const MapSemantic &semantic) noexcept {
  namespace lowering = rund::kernel::compute_lowering_detail;
  const std::uint32_t element_bytes =
      source.key.scalar == rund::kernel::ComputeScalar::Lane32
          ? sizeof(std::uint32_t)
          : source.key.scalar == rund::kernel::ComputeScalar::Lane64
                ? sizeof(std::uint64_t)
                : 0u;
  if (!device_vsm_typed_map::validate_total_scalar(
          source, input, semantic, source.key.scalar, source.key.domain,
          element_bytes) ||
      source.metadata.read_count == 0u ||
      source.metadata.read_count > DeviceVsmGraphStageInputCapacity ||
      source.metadata.write_count != 1u ||
      !source.metadata.param_storage.empty()) {
    return false;
  }
  std::uint32_t reads = 0u;
  for (const lowering::ParsedNode &node : input.parsed.nodes) {
    const auto op = static_cast<rund::kernel::IrOp>(node.op);
    if (op == rund::kernel::IrOp::Read) {
      ++reads;
    } else if (!device_vsm_typed_map::parameter_free_total_scalar_op_supported(
                   op)) {
      return false;
    }
  }
  return reads == source.metadata.read_count;
}

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise
