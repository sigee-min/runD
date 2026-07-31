#include "model.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] inline bool
SameFixedPolicy(const ComputeFixedFormat lhs,
                const ComputeFixedFormat rhs) noexcept {
  return lhs.rounding == rhs.rounding && lhs.overflow == rhs.overflow;
}

[[nodiscard]] inline bool
SameStoredFormat(const ComputeFixedFormat lhs,
                 const ComputeFixedFormat rhs) noexcept {
  return SameFixedPolicy(lhs, rhs) && lhs.integer_bits == rhs.integer_bits &&
         lhs.fraction_bits == rhs.fraction_bits;
}

[[nodiscard]] inline ComputeFixedFormat
DerivedBinaryFormat(const IrOp op, const ComputeFixedFormat lhs,
                    const ComputeFixedFormat rhs) noexcept {
  ComputeFixedFormat out = lhs;
  out.approximation =
      static_cast<u8>(lhs.approximation) >= static_cast<u8>(rhs.approximation)
          ? lhs.approximation
          : rhs.approximation;
  if (op == IrOp::Mul) {
    out.integer_bits =
        static_cast<u8>(static_cast<u32>(lhs.integer_bits) + rhs.integer_bits);
    out.fraction_bits = static_cast<u8>(static_cast<u32>(lhs.fraction_bits) +
                                        rhs.fraction_bits);
  } else if (op == IrOp::Add || op == IrOp::Sub) {
    out.integer_bits =
        static_cast<u8>(std::max<u32>(lhs.integer_bits, rhs.integer_bits) + 1u);
    out.fraction_bits =
        static_cast<u8>(std::max<u32>(lhs.fraction_bits, rhs.fraction_bits));
  } else {
    out.integer_bits =
        static_cast<u8>(std::max<u32>(lhs.integer_bits, rhs.integer_bits));
    out.fraction_bits =
        static_cast<u8>(std::max<u32>(lhs.fraction_bits, rhs.fraction_bits));
  }
  return out;
}

[[nodiscard]] const char *
ValidateFixedNodeFormat(const ParsedIR &parsed, const ParsedNode &node,
                        const ComputeScalar scalar) noexcept {
  if (parsed.scalar_mode != ScalarModeFor(scalar)) {
    if (static_cast<IrOp>(node.op) == IrOp::Write &&
        node.rhs == static_cast<u32>(IrWriteMode::BoundaryMask) &&
        node.aux < parsed.bindings.size()) {
      const ComputeDomain target =
          BindingDomainForShape(parsed.bindings[node.aux]);
      return target == ComputeDomain::Fixed
                 ? ComputeFixedFormatValid(scalar, node.fixed_format)
                       ? nullptr
                       : "compute_ir_numeric_policy_invalid"
             : ComputeFixedFormatAbsent(node.fixed_format)
                 ? nullptr
                 : "compute_ir_numeric_policy_mismatch";
    }
    return ComputeFixedFormatAbsent(node.fixed_format)
               ? nullptr
               : "compute_ir_numeric_policy_mismatch";
  }
  const auto op = static_cast<IrOp>(node.op);
  if (!ComputeIntermediateFormatValid(node.fixed_format)) {
    return "compute_ir_numeric_policy_invalid";
  }
  const auto source = [&](const u32 ref) -> const ComputeFixedFormat & {
    return parsed.nodes[ref - 1u].fixed_format;
  };
  switch (op) {
  case IrOp::Param:
  case IrOp::Read:
  case IrOp::ReadAt:
  case IrOp::Index:
    return ComputeFixedFormatValid(scalar, node.fixed_format)
               ? nullptr
               : "compute_ir_numeric_policy_invalid";
  case IrOp::Constant:
    return ComputeIntermediateFormatValid(node.fixed_format)
               ? nullptr
               : "compute_ir_numeric_policy_invalid";
  case IrOp::Write:
    return static_cast<IrOp>(parsed.nodes[node.lhs - 1u].op) ==
                       IrOp::Quantize &&
                   ComputeFixedFormatValid(scalar, source(node.lhs)) &&
                   node.fixed_format == source(node.lhs)
               ? nullptr
               : "compute_ir_quantize_required";
  case IrOp::Quantize: {
    return ComputeIntermediateFormatValid(source(node.lhs)) &&
                   ComputeFixedFormatValid(scalar, node.fixed_format) &&
                   !(source(node.lhs).approximation ==
                         ComputeApproximation::Deterministic &&
                     node.fixed_format.approximation !=
                         ComputeApproximation::Deterministic)
               ? nullptr
           : source(node.lhs).approximation ==
                       ComputeApproximation::Deterministic &&
                   node.fixed_format.approximation !=
                       ComputeApproximation::Deterministic
               ? "compute_ir_approximation_downgrade"
               : "compute_ir_quantize_policy_invalid";
  }
  case IrOp::Neg:
  case IrOp::Abs:
  case IrOp::AbsMagnitude: {
    auto expected = source(node.lhs);
    ++expected.integer_bits;
    return node.fixed_format == expected ? nullptr
                                         : "compute_ir_numeric_policy_mismatch";
  }
  case IrOp::Sign:
  case IrOp::PredicateNot:
    return node.fixed_format == source(node.lhs)
               ? nullptr
               : "compute_ir_numeric_policy_mismatch";
  case IrOp::BitNot:
  case IrOp::NegPositiveFixed:
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst:
    return ComputeFixedFormatValid(scalar, source(node.lhs)) &&
                   node.fixed_format == source(node.lhs)
               ? nullptr
               : "compute_ir_quantize_required";
  case IrOp::Recip:
  case IrOp::Sqrt:
  case IrOp::Rsqrt:
  case IrOp::Sin:
  case IrOp::Cos:
  case IrOp::Tan:
  case IrOp::Exp:
  case IrOp::Log: {
    auto expected = source(node.lhs);
    expected.approximation = ComputeApproximation::Deterministic;
    return ComputeFixedFormatValid(scalar, source(node.lhs)) &&
                   node.fixed_format == expected
               ? nullptr
               : "compute_ir_quantize_required";
  }
  case IrOp::Select: {
    const auto expected =
        DerivedBinaryFormat(IrOp::Min, source(node.rhs), source(node.aux));
    return SameFixedPolicy(source(node.rhs), source(node.aux)) &&
                   node.fixed_format == expected
               ? nullptr
               : "compute_ir_numeric_policy_mismatch";
  }
  case IrOp::Clamp: {
    auto expected =
        DerivedBinaryFormat(IrOp::Min, source(node.lhs), source(node.rhs));
    expected = DerivedBinaryFormat(IrOp::Min, expected, source(node.aux));
    return SameFixedPolicy(source(node.lhs), source(node.rhs)) &&
                   SameFixedPolicy(expected, source(node.aux)) &&
                   node.fixed_format == expected
               ? nullptr
               : "compute_ir_numeric_policy_mismatch";
  }
  case IrOp::MulAddFixed: {
    const auto &lhs = source(node.lhs);
    const auto &rhs = source(node.rhs);
    const auto &addend = source(node.aux);
    auto expected = lhs;
    const u32 product_integer =
        static_cast<u32>(lhs.integer_bits) + rhs.integer_bits;
    const u32 product_fraction =
        static_cast<u32>(lhs.fraction_bits) + rhs.fraction_bits;
    expected.integer_bits = static_cast<u8>(
        product_integer > addend.integer_bits ? product_integer
                                              : addend.integer_bits + 1u);
    expected.fraction_bits =
        static_cast<u8>(std::max<u32>(product_fraction, addend.fraction_bits));
    expected.approximation = static_cast<ComputeApproximation>(std::max(
        {static_cast<u8>(lhs.approximation), static_cast<u8>(rhs.approximation),
         static_cast<u8>(addend.approximation)}));
    return SameFixedPolicy(lhs, rhs) && SameFixedPolicy(lhs, addend) &&
                   node.fixed_format == expected
               ? nullptr
               : "compute_ir_numeric_policy_mismatch";
  }
  case IrOp::MulWrap:
  case IrOp::BitAnd:
  case IrOp::BitOr:
  case IrOp::BitXor: {
    auto expected = source(node.lhs);
    expected.approximation =
        static_cast<u8>(source(node.lhs).approximation) >=
                static_cast<u8>(source(node.rhs).approximation)
            ? source(node.lhs).approximation
            : source(node.rhs).approximation;
    return ComputeFixedFormatValid(scalar, source(node.lhs)) &&
                   ComputeFixedFormatValid(scalar, source(node.rhs)) &&
                   SameStoredFormat(source(node.lhs), source(node.rhs)) &&
                   node.fixed_format == expected
               ? nullptr
               : "compute_ir_quantize_required";
  }
  case IrOp::AddSat:
  case IrOp::AddSatUnsigned:
  case IrOp::SubSat:
  case IrOp::MulFixed:
  case IrOp::MulFixedScaled:
  case IrOp::MulUnsignedFixed: {
    auto expected = source(node.lhs);
    expected.approximation =
        static_cast<u8>(source(node.lhs).approximation) >=
                static_cast<u8>(source(node.rhs).approximation)
            ? source(node.lhs).approximation
            : source(node.rhs).approximation;
    return ComputeFixedFormatValid(scalar, source(node.lhs)) &&
                   ComputeFixedFormatValid(scalar, source(node.rhs)) &&
                   SameStoredFormat(source(node.lhs), source(node.rhs)) &&
                   node.fixed_format == expected
               ? nullptr
               : "compute_ir_quantize_required";
  }
  case IrOp::DivFixed:
  case IrOp::Atan2: {
    auto expected = source(node.lhs);
    expected.approximation = ComputeApproximation::Deterministic;
    return ComputeFixedFormatValid(scalar, source(node.lhs)) &&
                   ComputeFixedFormatValid(scalar, source(node.rhs)) &&
                   SameStoredFormat(source(node.lhs), source(node.rhs)) &&
                   node.fixed_format == expected
               ? nullptr
               : "compute_ir_quantize_required";
  }
  case IrOp::DivSigned:
  case IrOp::DivUnsigned:
    return "compute_ir_node_invalid";
  default: {
    const auto &lhs = source(node.lhs);
    const auto &rhs = source(node.rhs);
    const auto expected = DerivedBinaryFormat(op, lhs, rhs);
    return SameFixedPolicy(lhs, rhs) && node.fixed_format == expected
               ? nullptr
               : "compute_ir_numeric_policy_mismatch";
  }
  }
}
} // namespace rund::kernel::compute_lowering_detail
