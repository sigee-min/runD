#include "local.hpp"

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] bool CanonicalMaskConstant(const ParsedIR &parsed,
                                         const u32 node_ref,
                                         const u32 expected_low_bits) noexcept {
  if (node_ref == 0u || node_ref > parsed.nodes.size()) {
    return false;
  }
  const ParsedNode &node = parsed.nodes[node_ref - 1u];
  return static_cast<IrOp>(node.op) == IrOp::Constant &&
         node.lhs == expected_low_bits && node.rhs == 0u && node.aux == 0u;
}

[[nodiscard]] bool CanonicalMaskSource(const ParsedIR &parsed, u32 source_ref,
                                       const bool fixed_mode) noexcept {
  if (source_ref == 0u || source_ref > parsed.nodes.size()) {
    return false;
  }
  const ParsedNode *source = &parsed.nodes[source_ref - 1u];
  if (fixed_mode) {
    if (static_cast<IrOp>(source->op) != IrOp::Quantize || source->lhs == 0u ||
        source->lhs > parsed.nodes.size()) {
      return false;
    }
    const ComputeFixedFormat stored_format = source->fixed_format;
    source_ref = source->lhs;
    source = &parsed.nodes[source_ref - 1u];
    if (source->fixed_format != stored_format) {
      return false;
    }
  }
  if (static_cast<IrOp>(source->op) != IrOp::Select ||
      !CanonicalMaskConstant(parsed, source->rhs, 1u) ||
      !CanonicalMaskConstant(parsed, source->aux, 0u)) {
    return false;
  }
  if (!fixed_mode) {
    return true;
  }
  return parsed.nodes[source->rhs - 1u].fixed_format == source->fixed_format &&
         parsed.nodes[source->aux - 1u].fixed_format == source->fixed_format;
}

[[nodiscard]] bool
WidthChangingWriteIsCanonicalMask(const ParsedIR &parsed,
                                  const ComputeScalar scalar) noexcept {
  const u32 scalar_bytes = ScalarBytes(scalar);
  u32 write_binding_count = 0u;
  u32 width_changing_binding_count = 0u;
  u32 width_changing_binding = 0u;
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if (binding.kind == kWriteBindingKind) {
      ++write_binding_count;
    }
    if (binding.element_bytes == scalar_bytes) {
      continue;
    }
    if (ReadAtIndexBinding(parsed, static_cast<u32>(index))) {
      continue;
    }
    ++width_changing_binding_count;
    width_changing_binding = static_cast<u32>(index);
    const u8 expected_mode =
        binding.element_bytes == sizeof(u32)
            ? DomainModeFor(ComputeScalar::Lane32, ComputeDomain::U32)
            : DomainModeFor(ComputeScalar::Lane64, ComputeDomain::U64);
    if (binding.kind != kWriteBindingKind ||
        binding.numeric_mode != expected_mode) {
      return false;
    }
  }
  if (width_changing_binding_count == 0u) {
    return true;
  }
  if (width_changing_binding_count != 1u || write_binding_count != 1u) {
    return false;
  }

  u32 write_node_count = 0u;
  const ParsedNode *write = nullptr;
  for (const ParsedNode &node : parsed.nodes) {
    if (static_cast<IrOp>(node.op) != IrOp::Write) {
      continue;
    }
    ++write_node_count;
    write = &node;
  }
  return write_node_count == 1u && write != nullptr &&
         write->aux == width_changing_binding &&
         write->rhs == static_cast<u32>(IrWriteMode::Value) &&
         CanonicalMaskSource(parsed, write->lhs,
                             parsed.scalar_mode == ScalarModeFor(scalar));
}

[[nodiscard]] bool
CanonicalUnsignedMaskWriteBinding(const ParsedIR &parsed,
                                  const u32 binding) noexcept {
  bool observed = false;
  for (const ParsedNode &node : parsed.nodes) {
    if (static_cast<IrOp>(node.op) != IrOp::Write || node.aux != binding) {
      continue;
    }
    observed = true;
    if (node.rhs != static_cast<u32>(IrWriteMode::Value) ||
        !CanonicalMaskSource(parsed, node.lhs, true)) {
      return false;
    }
  }
  return observed;
}

} // namespace rund::kernel::compute_lowering_detail
