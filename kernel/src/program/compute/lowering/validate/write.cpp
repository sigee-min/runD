#include "local.hpp"

#include <limits>

namespace rund::kernel::compute_lowering_detail {
namespace {

[[nodiscard]] bool IntegerDomain(const ComputeDomain domain) noexcept {
  return domain == ComputeDomain::I32 || domain == ComputeDomain::U32 ||
         domain == ComputeDomain::I64 || domain == ComputeDomain::U64;
}

[[nodiscard]] bool NodeFormatAbsent(const ParsedNode &node) noexcept {
  return ComputeFixedFormatAbsent(node.fixed_format);
}

[[nodiscard]] bool
CheckedOrdinalWriteValid(const ParsedIR &parsed, const ParsedNode &write,
                         const ComputeScalar scalar,
                         const EffectiveNodeDomains &domains) noexcept {
  if (static_cast<IrOp>(write.op) != IrOp::Write ||
      write.rhs != static_cast<u32>(IrWriteMode::CheckedOrdinal) ||
      write.lhs == 0u || write.lhs > parsed.nodes.size() ||
      write.aux >= parsed.bindings.size() || !NodeFormatAbsent(write)) {
    return false;
  }
  const ParsedNode &select = parsed.nodes[write.lhs - 1u];
  if (static_cast<IrOp>(select.op) != IrOp::Select ||
      !NodeFormatAbsent(select) || select.lhs == 0u ||
      select.lhs > parsed.nodes.size() || select.rhs == 0u ||
      select.rhs > parsed.nodes.size() ||
      !CanonicalMaskConstant(parsed, select.aux, 0u) ||
      !NodeFormatAbsent(parsed.nodes[select.aux - 1u])) {
    return false;
  }
  const ParsedBinding &write_binding = parsed.bindings[write.aux];
  const ComputeDomain source_domain = domains.values[select.rhs - 1u];
  const ComputeDomain target_domain = BindingDomainForShape(write_binding);
  if (write_binding.kind != kWriteBindingKind ||
      write_binding.element_bytes != ScalarBytes(scalar) ||
      !IntegerDomain(source_domain) || !IntegerDomain(target_domain)) {
    return false;
  }
  const ParsedNode &predicate = parsed.nodes[select.lhs - 1u];
  if (!NodeFormatAbsent(predicate) || predicate.lhs != select.rhs ||
      predicate.rhs == 0u || predicate.rhs > parsed.nodes.size()) {
    return false;
  }
  const bool signed_to_unsigned = (source_domain == ComputeDomain::I32 &&
                                   target_domain == ComputeDomain::U32) ||
                                  (source_domain == ComputeDomain::I64 &&
                                   target_domain == ComputeDomain::U64);
  if (signed_to_unsigned) {
    return static_cast<IrOp>(predicate.op) == IrOp::Ge &&
           predicate.rhs == select.aux;
  }
  const bool unsigned_to_signed = (source_domain == ComputeDomain::U32 &&
                                   target_domain == ComputeDomain::I32) ||
                                  (source_domain == ComputeDomain::U64 &&
                                   target_domain == ComputeDomain::I64);
  if (!unsigned_to_signed) {
    return false;
  }
  const u64 expected = source_domain == ComputeDomain::U64
                           ? static_cast<u64>(std::numeric_limits<i64>::max())
                           : static_cast<u64>(std::numeric_limits<i32>::max());
  const ParsedNode &limit = parsed.nodes[predicate.rhs - 1u];
  const u64 bits =
      static_cast<u64>(limit.lhs) | (static_cast<u64>(limit.rhs) << 32u);
  return static_cast<IrOp>(predicate.op) == IrOp::LeUnsigned &&
         static_cast<IrOp>(limit.op) == IrOp::Constant && limit.aux == 0u &&
         NodeFormatAbsent(limit) && bits == expected;
}

[[nodiscard]] bool
BoundaryMaskWriteValid(const ParsedIR &parsed, const ParsedNode &write,
                       const ComputeScalar scalar,
                       const EffectiveNodeDomains &domains) noexcept {
  if (static_cast<IrOp>(write.op) != IrOp::Write ||
      write.rhs != static_cast<u32>(IrWriteMode::BoundaryMask) ||
      write.lhs == 0u || write.lhs > parsed.nodes.size() ||
      write.aux >= parsed.bindings.size()) {
    return false;
  }
  const ParsedBinding &target_binding = parsed.bindings[write.aux];
  const ComputeDomain target_domain = BindingDomainForShape(target_binding);
  const bool signed_target = target_domain == ComputeDomain::I32 ||
                             target_domain == ComputeDomain::I64;
  if (target_binding.kind != kWriteBindingKind ||
      target_binding.element_bytes != ScalarBytes(scalar) ||
      (!signed_target && target_domain != ComputeDomain::Fixed)) {
    return false;
  }

  const ParsedNode &select = parsed.nodes[write.lhs - 1u];
  if (static_cast<IrOp>(select.op) != IrOp::Select ||
      !NodeFormatAbsent(select) || select.lhs == 0u ||
      select.lhs > parsed.nodes.size() ||
      !CanonicalMaskConstant(parsed, select.rhs, 1u) ||
      !CanonicalMaskConstant(parsed, select.aux, 0u) ||
      !NodeFormatAbsent(parsed.nodes[select.rhs - 1u]) ||
      !NodeFormatAbsent(parsed.nodes[select.aux - 1u])) {
    return false;
  }
  const ParsedNode &predicate = parsed.nodes[select.lhs - 1u];
  if (static_cast<IrOp>(predicate.op) != IrOp::Ne ||
      !NodeFormatAbsent(predicate) || predicate.lhs == 0u ||
      predicate.lhs > parsed.nodes.size() ||
      !CanonicalMaskConstant(parsed, predicate.rhs, 0u) ||
      predicate.rhs != select.aux ||
      !NodeFormatAbsent(parsed.nodes[predicate.rhs - 1u])) {
    return false;
  }
  const ComputeDomain source_domain = domains.values[predicate.lhs - 1u];
  return IntegerDomain(source_domain) && source_domain != target_domain;
}

} // namespace

[[nodiscard]] const char *
ValidateWriteModes(const ParsedIR &parsed, const ComputeScalar scalar,
                   const EffectiveNodeDomains &domains) noexcept {
  for (const ParsedNode &node : parsed.nodes) {
    if (static_cast<IrOp>(node.op) != IrOp::Write) {
      continue;
    }
    switch (static_cast<IrWriteMode>(node.rhs)) {
    case IrWriteMode::Value: {
      ParsedNode boundary_candidate = node;
      boundary_candidate.rhs = static_cast<u32>(IrWriteMode::BoundaryMask);
      if (BoundaryMaskWriteValid(parsed, boundary_candidate, scalar, domains)) {
        return "compute_ir_node_invalid";
      }
      break;
    }
    case IrWriteMode::CheckedOrdinal:
      if (!CheckedOrdinalWriteValid(parsed, node, scalar, domains)) {
        return "compute_ir_node_invalid";
      }
      break;
    case IrWriteMode::BoundaryMask:
      if (!BoundaryMaskWriteValid(parsed, node, scalar, domains)) {
        return "compute_ir_node_invalid";
      }
      break;
    default:
      return "compute_ir_node_invalid";
    }
  }
  return nullptr;
}

} // namespace rund::kernel::compute_lowering_detail
