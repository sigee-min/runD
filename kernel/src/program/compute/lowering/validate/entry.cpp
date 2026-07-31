#include "model.hpp"

#include <vector>

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] const char *ValidateLowerableIR(const ParsedIR &parsed,
                                              const ComputeScalar scalar) {
  const bool fixed_mode = parsed.scalar_mode == ScalarModeFor(scalar);
  if ((fixed_mode && !ComputeFixedFormatValid(scalar, parsed.fixed_format)) ||
      (!fixed_mode && !ComputeFixedFormatAbsent(parsed.fixed_format))) {
    return "compute_ir_numeric_policy_mismatch";
  }
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if ((!BindingWidthShapeValid(binding.kind, binding.element_bytes, scalar) &&
         !ReadAtIndexBinding(parsed, static_cast<u32>(index))) ||
        BindingDomainForShape(binding) == static_cast<ComputeDomain>(0u)) {
      return "compute_ir_binding_scalar_mismatch";
    }
  }
  if (!BindingDomainsMatchGraph(parsed, scalar)) {
    return "compute_ir_binding_scalar_mismatch";
  }
  if (!WidthChangingWriteIsCanonicalMask(parsed, scalar)) {
    return "compute_ir_binding_scalar_mismatch";
  }

  const EffectiveNodeDomains effective_domains =
      ResolveEffectiveNodeDomains(parsed, scalar);
  if (!effective_domains.valid) {
    return "compute_ir_node_invalid";
  }
  if (const char *const reason =
          ValidateWriteModes(parsed, scalar, effective_domains);
      reason != nullptr) {
    return reason;
  }

  std::vector<bool> produces_value(parsed.nodes.size() + 1u, false);
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const ParsedNode &node = parsed.nodes[index];
    const u32 current_node = static_cast<u32>(index + 1u);
    const auto op = static_cast<IrOp>(node.op);
    const auto domain = effective_domains.values[index];
    if (CanonicalIrOpForDomain(op, domain) != op ||
        !IrOpDomainValid(op, domain)) {
      return "compute_ir_node_invalid";
    }
    if (const char *const reason =
            ValidateFixedNodeFormat(parsed, node, scalar);
        reason != nullptr) {
      return reason;
    }
    switch (static_cast<IrOp>(node.op)) {
    case IrOp::Param:
    case IrOp::Read:
    case IrOp::ReadAt:
    case IrOp::Constant:
    case IrOp::Index:
      produces_value[current_node] = true;
      break;
    case IrOp::Neg:
    case IrOp::Abs:
    case IrOp::AbsMagnitude:
    case IrOp::Sign:
    case IrOp::PredicateNot:
    case IrOp::BitNot:
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
      if (!produces_value[node.lhs]) {
        return "compute_ir_node_invalid";
      }
      produces_value[current_node] = true;
      break;
    case IrOp::Add:
    case IrOp::Sub:
    case IrOp::Mul:
    case IrOp::MulWrap:
    case IrOp::Min:
    case IrOp::Max:
    case IrOp::Eq:
    case IrOp::Lt:
    case IrOp::Le:
    case IrOp::Ne:
    case IrOp::Gt:
    case IrOp::Ge:
    case IrOp::PredicateAnd:
    case IrOp::PredicateOr:
    case IrOp::BitAnd:
    case IrOp::BitOr:
    case IrOp::BitXor:
    case IrOp::AddSat:
    case IrOp::AddSatUnsigned:
    case IrOp::SubSat:
    case IrOp::MulFixed:
    case IrOp::MulFixedScaled:
    case IrOp::MulUnsignedFixed:
    case IrOp::DivFixed:
    case IrOp::Atan2:
    case IrOp::DivSigned:
    case IrOp::DivUnsigned:
    case IrOp::MinUnsigned:
    case IrOp::MaxUnsigned:
    case IrOp::LtUnsigned:
    case IrOp::LeUnsigned:
    case IrOp::GtUnsigned:
    case IrOp::GeUnsigned:
      if (!produces_value[node.lhs] || !produces_value[node.rhs]) {
        return "compute_ir_node_invalid";
      }
      produces_value[current_node] = true;
      break;
    case IrOp::ShlConst:
    case IrOp::ShrLogicalConst:
    case IrOp::ShrArithmeticConst:
      if (!produces_value[node.lhs]) {
        return "compute_ir_node_invalid";
      }
      if (node.aux >= ScalarBitWidth(scalar)) {
        return "compute_shift_count_invalid";
      }
      produces_value[current_node] = true;
      break;
    case IrOp::Clamp:
    case IrOp::ClampUnsigned:
    case IrOp::Select:
    case IrOp::MulAddFixed:
      if (!produces_value[node.lhs] || !produces_value[node.rhs] ||
          !produces_value[node.aux]) {
        return "compute_ir_node_invalid";
      }
      produces_value[current_node] = true;
      break;
    case IrOp::Write:
      if (!produces_value[node.lhs]) {
        return "compute_ir_node_invalid";
      }
      break;
    }
  }
  return nullptr;
}
} // namespace rund::kernel::compute_lowering_detail
