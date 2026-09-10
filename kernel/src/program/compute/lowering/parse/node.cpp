#include "local.hpp"

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] ParsedNodeRead ReadParsedNode(Reader &reader) {
  ParsedNode node{};
  u8 rounding = 0u;
  u8 overflow = 0u;
  u8 approximation = 0u;
  if (!reader.read_u8(node.op) || !reader.read_u32(node.lhs) ||
      !reader.read_u32(node.rhs) || !reader.read_u32(node.aux) ||
      !reader.read_u8(node.fixed_format.integer_bits) ||
      !reader.read_u8(node.fixed_format.fraction_bits) ||
      !reader.read_u8(rounding) || !reader.read_u8(overflow) ||
      !reader.read_u8(approximation)) {
    return ParsedNodeRead{};
  }
  node.fixed_format.rounding = static_cast<ComputeRounding>(rounding);
  node.fixed_format.overflow = static_cast<ComputeOverflow>(overflow);
  node.fixed_format.approximation =
      static_cast<ComputeApproximation>(approximation);
  if (OpName(node.op) == nullptr) {
    return ParsedNodeRead{.node = node, .reason = "compute_ir_op_unsupported"};
  }
  if (!ComputeIntermediateFormatValid(node.fixed_format)) {
    return ParsedNodeRead{.node = node,
                          .reason = "compute_ir_numeric_policy_invalid"};
  }
  return ParsedNodeRead{.node = node, .ok = true, .reason = "ok"};
}

[[nodiscard]] const char *
ValidateParsedNodeOperands(const ParsedIR &parsed, const ParsedNode &node,
                           const ComputeScalar scalar,
                           const u32 current_node) noexcept {
  switch (static_cast<IrOp>(node.op)) {
  case IrOp::Param:
    if (node.lhs != 0u || node.rhs != 0u ||
        !BindingIs(parsed, node.aux, kParamBindingKind)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::Read:
  case IrOp::ReadUniform:
    if (node.lhs != 0u || node.rhs != 0u ||
        !BindingIs(parsed, node.aux, kReadBindingKind)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::ReadAt:
    if (!BindingIs(parsed, node.lhs, kReadBindingKind) ||
        !BindingIs(parsed, node.aux, kReadBindingKind) ||
        node.lhs == node.aux || node.rhs == 0u ||
        parsed.bindings[node.lhs].numeric_mode != 4u ||
        parsed.bindings[node.lhs].element_bytes != sizeof(u32)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::Constant:
    if (node.aux != 0u || (scalar == ComputeScalar::Lane32 && node.rhs != 0u)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::Index:
    if (node.lhs != 0u || node.rhs != 0u || node.aux != 0u) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::Write:
    if (!ValidNodeRef(node.lhs, current_node) || !IrWriteModeValid(node.rhs) ||
        !BindingIs(parsed, node.aux, kWriteBindingKind)) {
      return "compute_ir_node_invalid";
    }
    if (parsed.scalar_mode == ScalarModeFor(scalar) &&
        node.rhs != static_cast<u32>(IrWriteMode::Value)) {
      return "compute_ir_node_invalid";
    }
    if (parsed.scalar_mode == ScalarModeFor(scalar) &&
        static_cast<IrOp>(parsed.nodes[node.lhs - 1u].op) != IrOp::Quantize) {
      return "compute_ir_quantize_required";
    }
    return nullptr;
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
    if (!ValidNodeRef(node.lhs, current_node) || node.rhs != 0u ||
        node.aux != 0u) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
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
    if (!ValidNodeRef(node.lhs, current_node) ||
        !ValidNodeRef(node.rhs, current_node) || node.aux != 0u) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst:
    if (!ValidNodeRef(node.lhs, current_node) || node.rhs != 0u) {
      return "compute_ir_node_invalid";
    }
    return node.aux >= ScalarBitWidth(scalar) ? "compute_shift_count_invalid"
                                              : nullptr;
  case IrOp::Clamp:
  case IrOp::ClampUnsigned:
  case IrOp::Select:
  case IrOp::MulAddFixed:
    if (!ValidNodeRef(node.lhs, current_node) ||
        !ValidNodeRef(node.rhs, current_node) ||
        !ValidNodeRef(node.aux, current_node)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  }
  return "compute_ir_op_unsupported";
}

[[nodiscard]] const char *AppendParsedNode(Reader &reader, ParsedIR &parsed,
                                           const ComputeScalar scalar,
                                           const u32 index, u32 &write_count) {
  ParsedNodeRead read = ReadParsedNode(reader);
  if (!read.ok) {
    return read.reason;
  }
  const bool nonfixed_boundary_format =
      static_cast<IrOp>(read.node.op) == IrOp::Write &&
      read.node.rhs == static_cast<u32>(IrWriteMode::BoundaryMask);
  if (parsed.scalar_mode != ScalarModeFor(scalar) &&
      !ComputeFixedFormatAbsent(read.node.fixed_format) &&
      !nonfixed_boundary_format) {
    return "compute_ir_numeric_policy_mismatch";
  }
  const u32 current_node = index + 1u;
  if (const char *const reason =
          ValidateParsedNodeOperands(parsed, read.node, scalar, current_node);
      reason != nullptr) {
    return reason;
  }
  if (static_cast<IrOp>(read.node.op) == IrOp::Write) {
    ++write_count;
  }
  parsed.nodes.push_back(read.node);
  return nullptr;
}

} // namespace rund::kernel::compute_lowering_detail
