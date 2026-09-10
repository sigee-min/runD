#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_typed_map {

bool validate_total_scalar(
    const rund::kernel::LoweringArtifact &source,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const MapSemantic &semantic, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ComputeDomain domain,
    const std::uint32_t element_bytes) noexcept {
  namespace lowering = rund::kernel::compute_lowering_detail;
  if (!((scalar == rund::kernel::ComputeScalar::Lane32 &&
         domain == rund::kernel::ComputeDomain::U32 && element_bytes == 4u) ||
        (scalar == rund::kernel::ComputeScalar::Lane64 &&
         domain == rund::kernel::ComputeDomain::U64 && element_bytes == 8u))) {
    return false;
  }
  const std::uint8_t mode = lowering::DomainModeFor(scalar, domain);
  if (!source.ok || source.key.scalar != scalar ||
      source.key.domain != domain ||
      source.key.variant != rund::kernel::LoweringArtifactVariant::Canonical ||
      source.key.fixed_format != rund::kernel::ComputeFixedFormat{} ||
      source.metadata.read_count == 0u || source.metadata.write_count != 1u ||
      !source.metadata.read_routes.empty() || !input.ok ||
      input.key != source.key || !input.parsed.ok ||
      input.parsed.scalar_mode != mode ||
      input.parsed.fixed_format != source.key.fixed_format ||
      !semantic.recurrence_total) {
    return false;
  }
  std::uint32_t reads = 0u;
  std::uint32_t writes = 0u;
  for (const lowering::ParsedBinding &binding : input.parsed.bindings) {
    if (binding.kind == lowering::kReadBindingKind) {
      reads += 1u;
      if (binding.numeric_mode != mode ||
          binding.element_bytes != element_bytes) {
        return false;
      }
    } else if (binding.kind == lowering::kWriteBindingKind) {
      writes += 1u;
      if (binding.numeric_mode != mode ||
          binding.element_bytes != element_bytes) {
        return false;
      }
    } else {
      return false;
    }
  }
  if (reads != source.metadata.read_count || writes != 1u ||
      input.parsed.nodes.empty()) {
    return false;
  }
  std::uint32_t write_nodes = 0u;
  for (std::size_t index = 0u; index < input.parsed.nodes.size(); ++index) {
    const lowering::ParsedNode &node = input.parsed.nodes[index];
    if (node.op != static_cast<std::uint8_t>(rund::kernel::IrOp::Write)) {
      continue;
    }
    write_nodes += 1u;
    if (index + 1u != input.parsed.nodes.size() || node.lhs == 0u ||
        node.rhs !=
            static_cast<std::uint32_t>(rund::kernel::IrWriteMode::Value) ||
        node.aux >= input.parsed.bindings.size() ||
        input.parsed.bindings[node.aux].kind != lowering::kWriteBindingKind) {
      return false;
    }
  }
  return write_nodes == 1u;
}

bool validate_total_u64(
    const rund::kernel::LoweringArtifact &source,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const MapSemantic &semantic) noexcept {
  return validate_total_scalar(
      source, input, semantic, rund::kernel::ComputeScalar::Lane64,
      rund::kernel::ComputeDomain::U64, sizeof(std::uint64_t));
}

} // namespace rund::node::accel::detail::device_vsm_typed_map
