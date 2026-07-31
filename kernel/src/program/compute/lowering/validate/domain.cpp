#include "model.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] inline bool
CanonicalMaskConstant(const ParsedIR &parsed, const u32 node_ref,
                      const u32 expected_low_bits) noexcept {
  if (node_ref == 0u || node_ref > parsed.nodes.size()) {
    return false;
  }
  const ParsedNode &node = parsed.nodes[node_ref - 1u];
  return static_cast<IrOp>(node.op) == IrOp::Constant &&
         node.lhs == expected_low_bits && node.rhs == 0u && node.aux == 0u;
}

[[nodiscard]] inline bool CanonicalMaskSource(const ParsedIR &parsed,
                                              u32 source_ref,
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

[[nodiscard]] ComputeDomain
BindingDomainForShape(const ParsedBinding &binding) noexcept {
  const ComputeScalar binding_scalar =
      binding.element_bytes == sizeof(u32)   ? ComputeScalar::Lane32
      : binding.element_bytes == sizeof(u64) ? ComputeScalar::Lane64
                                             : static_cast<ComputeScalar>(0u);
  return binding_scalar == static_cast<ComputeScalar>(0u)
             ? static_cast<ComputeDomain>(0u)
             : DomainForMode(binding_scalar, binding.numeric_mode);
}

[[nodiscard]] inline bool
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

[[nodiscard]] inline bool IntegerDomain(const ComputeDomain domain) noexcept {
  return domain == ComputeDomain::I32 || domain == ComputeDomain::U32 ||
         domain == ComputeDomain::I64 || domain == ComputeDomain::U64;
}

[[nodiscard]] inline bool NodeFormatAbsent(const ParsedNode &node) noexcept {
  return ComputeFixedFormatAbsent(node.fixed_format);
}

[[nodiscard]] bool ReadAtIndexBinding(const ParsedIR &parsed,
                                      const u32 binding) noexcept {
  bool indexed = false;
  for (const ParsedNode &node : parsed.nodes) {
    const auto op = static_cast<IrOp>(node.op);
    if (op == IrOp::ReadAt && node.lhs == binding) {
      indexed = true;
    }
    if ((op == IrOp::Read && node.aux == binding) ||
        (op == IrOp::ReadAt && node.aux == binding)) {
      return false;
    }
  }
  return indexed;
}

[[nodiscard]] inline bool
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

[[nodiscard]] inline bool
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

[[nodiscard]] inline bool
BoundaryMaskWriteBinding(const ParsedIR &parsed, const u32 binding,
                         const ComputeScalar) noexcept {
  bool observed = false;
  for (const ParsedNode &node : parsed.nodes) {
    if (static_cast<IrOp>(node.op) != IrOp::Write || node.aux != binding) {
      continue;
    }
    observed = true;
    if (node.rhs != static_cast<u32>(IrWriteMode::BoundaryMask)) {
      return false;
    }
  }
  return observed;
}

[[nodiscard]] bool
BindingDomainsMatchGraph(const ParsedIR &parsed,
                         const ComputeScalar scalar) noexcept {
  const bool fixed_mode = parsed.scalar_mode == ScalarModeFor(scalar);
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    const ComputeDomain binding_domain = BindingDomainForShape(binding);
    if (binding_domain == static_cast<ComputeDomain>(0u)) {
      return false;
    }
    if (ReadAtIndexBinding(parsed, static_cast<u32>(index))) {
      continue;
    }
    if (!fixed_mode) {
      if (binding_domain == ComputeDomain::Fixed) {
        if (binding.kind != kWriteBindingKind ||
            !BoundaryMaskWriteBinding(parsed, static_cast<u32>(index),
                                      scalar)) {
          return false;
        }
      }
      continue;
    }
    if (binding_domain == ComputeDomain::Fixed) {
      continue;
    }
    if (binding.kind != kWriteBindingKind || !UnsignedDomain(binding_domain) ||
        !CanonicalUnsignedMaskWriteBinding(parsed, static_cast<u32>(index))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] EffectiveNodeDomains
ResolveEffectiveNodeDomains(const ParsedIR &parsed,
                            const ComputeScalar scalar) {
  const auto unknown = static_cast<ComputeDomain>(0u);
  const bool fixed_mode = parsed.scalar_mode == ScalarModeFor(scalar);
  EffectiveNodeDomains resolved{
      .values = std::vector<ComputeDomain>(
          parsed.nodes.size(), fixed_mode ? ComputeDomain::Fixed : unknown),
      .valid = true,
  };
  if (fixed_mode) {
    return resolved;
  }

  const auto binding_domain = [&](const u32 binding) {
    if (binding >= parsed.bindings.size()) {
      return unknown;
    }
    return BindingDomainForShape(parsed.bindings[binding]);
  };
  std::vector<bool> canonical_value_mask_results(parsed.nodes.size() + 1u,
                                                 false);
  for (const ParsedNode &node : parsed.nodes) {
    if (static_cast<IrOp>(node.op) != IrOp::Write ||
        node.rhs != static_cast<u32>(IrWriteMode::Value) || node.lhs == 0u ||
        node.lhs > parsed.nodes.size()) {
      continue;
    }
    const ComputeDomain target_domain = binding_domain(node.aux);
    if ((target_domain == ComputeDomain::U32 ||
         target_domain == ComputeDomain::U64) &&
        CanonicalMaskSource(parsed, node.lhs, false)) {
      canonical_value_mask_results[node.lhs] = true;
    }
  }
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const ParsedNode &node = parsed.nodes[index];
    const auto op = static_cast<IrOp>(node.op);
    if (op == IrOp::Param || op == IrOp::Read || op == IrOp::ReadAt) {
      resolved.values[index] = binding_domain(node.aux);
      if (resolved.values[index] == unknown ||
          resolved.values[index] == ComputeDomain::Fixed) {
        resolved.valid = false;
      }
      continue;
    }
    if (op == IrOp::Write) {
      const auto mode = static_cast<IrWriteMode>(node.rhs);
      resolved.values[index] = mode == IrWriteMode::BoundaryMask
                                   ? DomainForMode(scalar, parsed.scalar_mode)
                                   : binding_domain(node.aux);
      if (resolved.values[index] == unknown ||
          resolved.values[index] == ComputeDomain::Fixed) {
        resolved.valid = false;
      }
      continue;
    }
    const ComputeDomain signed_domain = scalar == ComputeScalar::Lane64
                                            ? ComputeDomain::I64
                                            : ComputeDomain::I32;
    const ComputeDomain unsigned_domain = scalar == ComputeScalar::Lane64
                                              ? ComputeDomain::U64
                                              : ComputeDomain::U32;
    const bool signed_only = IrOpDomainValid(op, signed_domain) &&
                             !IrOpDomainValid(op, unsigned_domain);
    const bool unsigned_only = IrOpDomainValid(op, unsigned_domain) &&
                               !IrOpDomainValid(op, signed_domain);
    if (signed_only) {
      resolved.values[index] = signed_domain;
    } else if (unsigned_only) {
      resolved.values[index] = unsigned_domain;
    }
  }

  const auto assign = [&](const u32 reference, const ComputeDomain expected) {
    if (reference == 0u || reference > resolved.values.size()) {
      resolved.valid = false;
      return false;
    }
    if (expected == unknown) {
      return false;
    }
    ComputeDomain &current = resolved.values[reference - 1u];
    if (current == unknown) {
      current = expected;
      return true;
    }
    if (current != expected) {
      resolved.valid = false;
    }
    return false;
  };

  const auto constrain_unary = [&](const u32 result, const u32 source) {
    if (result == 0u || result > resolved.values.size() || source == 0u ||
        source > resolved.values.size()) {
      resolved.valid = false;
      return false;
    }
    const ComputeDomain result_domain = resolved.values[result - 1u];
    const ComputeDomain source_domain = resolved.values[source - 1u];
    bool changed = false;
    if (result_domain != unknown) {
      changed = assign(source, result_domain) || changed;
    }
    if (source_domain != unknown) {
      changed = assign(result, source_domain) || changed;
    }
    return changed;
  };

  const auto constrain_binary = [&](const u32 result, const u32 lhs,
                                    const u32 rhs) {
    if (result == 0u || result > resolved.values.size() || lhs == 0u ||
        lhs > resolved.values.size() || rhs == 0u ||
        rhs > resolved.values.size()) {
      resolved.valid = false;
      return false;
    }
    ComputeDomain left = resolved.values[lhs - 1u];
    ComputeDomain right = resolved.values[rhs - 1u];
    ComputeDomain result_domain = resolved.values[result - 1u];
    ComputeDomain merged = MergeComputeDomains(left, right);
    bool changed = false;
    if (left != unknown && right != unknown && merged == unknown) {
      resolved.valid = false;
      return false;
    }
    if (merged != unknown) {
      changed = assign(result, merged) || changed;
      result_domain = resolved.values[result - 1u];
    }
    if (result_domain == unknown) {
      return changed;
    }
    if (left == unknown) {
      changed = assign(lhs, result_domain) || changed;
      left = resolved.values[lhs - 1u];
    }
    if (right == unknown) {
      changed = assign(rhs, result_domain) || changed;
      right = resolved.values[rhs - 1u];
    }
    merged = MergeComputeDomains(left, right);
    if (merged == unknown || merged != result_domain) {
      resolved.valid = false;
    }
    return changed;
  };

  const auto constrain_node = [&](const std::size_t index) {
    if (index >= parsed.nodes.size() || !resolved.valid) {
      if (index >= parsed.nodes.size()) {
        resolved.valid = false;
      }
      return;
    }
    const ParsedNode &node = parsed.nodes[index];
    const u32 result = static_cast<u32>(index + 1u);
    switch (static_cast<IrOp>(node.op)) {
    case IrOp::Param:
    case IrOp::Read:
    case IrOp::ReadAt:
    case IrOp::Constant:
    case IrOp::Index:
      break;
    case IrOp::Write: {
      const ComputeDomain target_domain = binding_domain(node.aux);
      const bool target_unsigned = target_domain == ComputeDomain::U32 ||
                                   target_domain == ComputeDomain::U64;
      if (node.rhs == static_cast<u32>(IrWriteMode::Value) &&
          !(target_unsigned && CanonicalMaskSource(parsed, node.lhs, false))) {
        (void)assign(node.lhs, target_domain);
      }
      break;
    }
    case IrOp::Neg:
    case IrOp::Abs:
    case IrOp::AbsMagnitude:
    case IrOp::Sign:
    case IrOp::PredicateNot:
    case IrOp::BitNot:
    case IrOp::ShlConst:
    case IrOp::ShrLogicalConst:
    case IrOp::ShrArithmeticConst:
    case IrOp::NegPositiveFixed:
    case IrOp::Recip:
    case IrOp::Sqrt:
    case IrOp::Rsqrt:
    case IrOp::Sin:
    case IrOp::Cos:
    case IrOp::Tan:
    case IrOp::Exp:
    case IrOp::Log:
    case IrOp::Quantize:
      (void)constrain_unary(result, node.lhs);
      break;
    case IrOp::Select:
      (void)constrain_binary(result, node.rhs, node.aux);
      if (resolved.values[result - 1u] == unknown &&
          canonical_value_mask_results[result]) {
        (void)constrain_unary(result, node.lhs);
      }
      break;
    case IrOp::Clamp:
    case IrOp::ClampUnsigned:
    case IrOp::MulAddFixed:
      (void)constrain_binary(result, node.lhs, node.rhs);
      (void)constrain_binary(result, result, node.aux);
      break;
    default:
      (void)constrain_binary(result, node.lhs, node.rhs);
      break;
    }
  };
  const auto sweep = [&](const bool reverse) {
    if (reverse) {
      for (std::size_t index = parsed.nodes.size();
           index > 0u && resolved.valid; --index) {
        constrain_node(index - 1u);
      }
      return;
    }
    for (std::size_t index = 0u; index < parsed.nodes.size() && resolved.valid;
         ++index) {
      constrain_node(index);
    }
  };

  sweep(true);
  sweep(false);
  const ComputeDomain header_domain = DomainForMode(scalar, parsed.scalar_mode);
  for (ComputeDomain &value_domain : resolved.values) {
    if (value_domain == unknown) {
      value_domain = header_domain;
    }
  }
  sweep(true);
  sweep(false);
  for (const ComputeDomain value_domain : resolved.values) {
    if (value_domain == unknown) {
      resolved.valid = false;
      break;
    }
  }
  return resolved;
}
} // namespace rund::kernel::compute_lowering_detail
