#include "local.hpp"

#include <vector>

namespace rund::kernel::compute_lowering_detail {

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
    if (op == IrOp::Param || op == IrOp::Read || op == IrOp::ReadUniform ||
        op == IrOp::ReadAt) {
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
    case IrOp::ReadUniform:
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
