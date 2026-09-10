#include "../scalar.hpp"

namespace rund::node::accel::detail::device_vsm_typed_map {
bool parameter_free_total_scalar_op_supported(
    const rund::kernel::IrOp op) noexcept {
  using rund::kernel::IrOp;
  switch (op) {
  case IrOp::Read:
  case IrOp::Write:
  case IrOp::Add:
  case IrOp::Sub:
  case IrOp::Mul:
  case IrOp::MulWrap:
  case IrOp::Min:
  case IrOp::Max:
  case IrOp::Clamp:
  case IrOp::Select:
  case IrOp::Eq:
  case IrOp::Lt:
  case IrOp::Le:
  case IrOp::Constant:
  case IrOp::Neg:
  case IrOp::Abs:
  case IrOp::AbsMagnitude:
  case IrOp::Sign:
  case IrOp::Ne:
  case IrOp::Gt:
  case IrOp::Ge:
  case IrOp::PredicateNot:
  case IrOp::PredicateAnd:
  case IrOp::PredicateOr:
  case IrOp::BitAnd:
  case IrOp::BitOr:
  case IrOp::BitXor:
  case IrOp::BitNot:
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst:
  case IrOp::MinUnsigned:
  case IrOp::MaxUnsigned:
  case IrOp::ClampUnsigned:
  case IrOp::AddSatUnsigned:
  case IrOp::LtUnsigned:
  case IrOp::LeUnsigned:
  case IrOp::GtUnsigned:
  case IrOp::GeUnsigned:
  case IrOp::Index:
  case IrOp::Quantize:
    return true;
  default:
    return false;
  }
}

bool validate_parameter_free_total_u32_scalar(
    const rund::kernel::LoweringArtifact &source,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission
        &input) noexcept {
  namespace lowering = rund::kernel::compute_lowering_detail;
  constexpr std::uint8_t U32Mode = lowering::DomainModeFor(
      rund::kernel::ComputeScalar::Lane32, rund::kernel::ComputeDomain::U32);
  if (!source.ok || source.key.scalar != rund::kernel::ComputeScalar::Lane32 ||
      source.key.domain != rund::kernel::ComputeDomain::U32 ||
      source.key.variant != rund::kernel::LoweringArtifactVariant::Canonical ||
      source.key.fixed_format != rund::kernel::ComputeFixedFormat{} ||
      source.metadata.read_count != 1u || source.metadata.write_count != 1u ||
      !source.metadata.read_routes.empty() || !input.ok ||
      input.key != source.key || !input.parsed.ok ||
      input.parsed.scalar_mode != U32Mode ||
      input.parsed.fixed_format != source.key.fixed_format ||
      input.parsed.nodes.empty()) {
    return false;
  }
  std::uint32_t reads = 0u;
  std::uint32_t writes = 0u;
  for (const lowering::ParsedBinding &binding : input.parsed.bindings) {
    if (binding.kind == lowering::kReadBindingKind) {
      reads += 1u;
    } else if (binding.kind == lowering::kWriteBindingKind) {
      writes += 1u;
    } else {
      return false;
    }
    if (binding.numeric_mode != U32Mode ||
        binding.element_bytes != sizeof(std::uint32_t)) {
      return false;
    }
  }
  if (reads != 1u || writes != 1u) {
    return false;
  }
  std::uint32_t write_nodes = 0u;
  for (std::size_t index = 0u; index < input.parsed.nodes.size(); ++index) {
    const lowering::ParsedNode &node = input.parsed.nodes[index];
    const auto op = static_cast<rund::kernel::IrOp>(node.op);
    if (!parameter_free_total_scalar_op_supported(op)) {
      return false;
    }
    if (op != rund::kernel::IrOp::Write) {
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

} // namespace rund::node::accel::detail::device_vsm_typed_map
