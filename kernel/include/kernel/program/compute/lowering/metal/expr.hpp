#pragma once

#include <kernel/program/compute/lowering/names.hpp>
#include <kernel/program/compute/lowering/metal/syntax.hpp>
#include <kernel/program/compute/lowering/text.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] inline std::string MetalWrapBinaryExpr(const ComputeScalar scalar,
                                                     const std::string &lhs,
                                                     const char *const op_text,
                                                     const std::string &rhs) {
  const char *const scalar_type = MetalType(scalar);
  const char *const unsigned_type = MetalUnsignedType(scalar);
  std::string expr = scalar_type;
  expr += "(";
  expr += unsigned_type;
  expr += "(";
  expr += lhs;
  expr += ") ";
  expr += op_text;
  expr += " ";
  expr += unsigned_type;
  expr += "(";
  expr += rhs;
  expr += "))";
  return expr;
}

[[nodiscard]] inline std::string HexLiteral(const ComputeScalar scalar,
                                            const u64 value) {
  std::string expr = "0x";
  if (scalar == ComputeScalar::Lane64) {
    AppendHex64Digits(expr, value);
    expr += "ul";
    return expr;
  }
  constexpr char kHex[] = "0123456789abcdef";
  const u32 narrow = static_cast<u32>(value);
  for (int shift = 28; shift >= 0; shift -= 4) {
    expr += kHex[(narrow >> static_cast<unsigned>(shift)) & 0xfu];
  }
  expr += "u";
  return expr;
}

[[nodiscard]] inline u64 HighMask(const ComputeScalar scalar,
                                  const u32 amount) noexcept {
  if (amount == 0u) {
    return 0u;
  }
  const u32 width = ScalarBitWidth(scalar);
  if (scalar == ComputeScalar::Lane64) {
    return ~u64{0u} << (width - amount);
  }
  return static_cast<u64>(~u32{0u} << (width - amount));
}

[[nodiscard]] inline std::string MetalBitBinaryExpr(const ComputeScalar scalar,
                                                    const std::string &lhs,
                                                    const char *const op_text,
                                                    const std::string &rhs) {
  std::string expr = MetalType(scalar);
  expr += "(";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr += lhs;
  expr += ") ";
  expr += op_text;
  expr += " ";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr += rhs;
  expr += "))";
  return expr;
}

[[nodiscard]] inline std::string MetalBitNotExpr(const ComputeScalar scalar,
                                                 const std::string &value) {
  std::string expr = MetalType(scalar);
  expr += "(~";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr += value;
  expr += "))";
  return expr;
}

[[nodiscard]] inline std::string MetalShiftExpr(const ComputeScalar scalar,
                                                const IrOp op,
                                                const std::string &value,
                                                const u32 amount) {
  std::string expr = MetalType(scalar);
  expr += "(";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr += value;
  expr += ") ";
  expr += op == IrOp::ShlConst ? "<< " : ">> ";
  expr += std::to_string(amount);
  expr += "u";
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
MetalArithmeticShiftExpr(const ComputeScalar scalar, const std::string &value,
                         const u32 amount) {
  std::string expr = MetalType(scalar);
  expr += "((";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr += value;
  expr += ") >> ";
  expr += std::to_string(amount);
  expr += "u) | ((";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr += value;
  expr += ") & ";
  expr += HexLiteral(scalar, scalar == ComputeScalar::Lane64
                                 ? 0x8000000000000000ull
                                 : 0x80000000ull);
  expr += ") != ";
  expr += MetalUnsignedType(scalar);
  expr += "(0) ? ";
  expr += HexLiteral(scalar, HighMask(scalar, amount));
  expr += " : ";
  expr += MetalUnsignedType(scalar);
  expr += "(0)))";
  return expr;
}

[[nodiscard]] inline std::string MetalCompareExpr(const ComputeDomain domain,
                                                  const ComputeScalar scalar,
                                                  const std::string &lhs,
                                                  const char *const op_text,
                                                  const std::string &rhs) {
  if (!UnsignedDomain(domain)) {
    return lhs + " " + op_text + " " + rhs;
  }
  const char *const type = MetalUnsignedType(scalar);
  return std::string{type} + "(" + lhs + ") " + op_text + " " + type + "(" +
         rhs + ")";
}

[[nodiscard]] inline std::string MetalPredicateExpr(const ComputeDomain domain,
                                                    const ComputeScalar scalar,
                                                    const std::string &lhs,
                                                    const char *const op_text,
                                                    const std::string &rhs) {
  const char *const scalar_type = MetalType(scalar);
  std::string expr = MetalCompareExpr(domain, scalar, lhs, op_text, rhs);
  expr += " ? ";
  expr += scalar_type;
  expr += "(1) : ";
  expr += scalar_type;
  expr += "(0)";
  return expr;
}

[[nodiscard]] inline std::string MetalMinExpr(const ComputeDomain domain,
                                              const ComputeScalar scalar,
                                              const std::string &lhs,
                                              const std::string &rhs) {
  return MetalCompareExpr(domain, scalar, lhs, "<", rhs) + " ? " + lhs + " : " +
         rhs;
}

[[nodiscard]] inline std::string MetalMaxExpr(const ComputeDomain domain,
                                              const ComputeScalar scalar,
                                              const std::string &lhs,
                                              const std::string &rhs) {
  return MetalCompareExpr(domain, scalar, lhs, ">", rhs) + " ? " + lhs + " : " +
         rhs;
}

[[nodiscard]] inline std::string MetalConstantExpr(const ComputeScalar scalar,
                                                   const u64 bits) {
  const char *const scalar_type = MetalType(scalar);
  const char *const unsigned_type = MetalUnsignedType(scalar);
  std::string expr = scalar_type;
  expr += "(";
  expr += unsigned_type;
  expr += "(";
  expr += std::to_string(bits);
  expr += scalar == ComputeScalar::Lane64 ? "ul" : "u";
  expr += "))";
  return expr;
}

[[nodiscard]] inline std::string MetalZeroExpr(const ComputeScalar scalar) {
  std::string expr = MetalType(scalar);
  expr += "(0)";
  return expr;
}

[[nodiscard]] inline std::string MetalOneExpr(const ComputeScalar scalar) {
  std::string expr = MetalType(scalar);
  expr += "(1)";
  return expr;
}

[[nodiscard]] inline std::string MetalMinusOneExpr(const ComputeScalar scalar) {
  std::string expr = MetalType(scalar);
  expr += "(";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr +=
      scalar == ComputeScalar::Lane64 ? "0xfffffffffffffffful" : "0xffffffffu";
  expr += "))";
  return expr;
}

[[nodiscard]] inline std::string MetalFixedMinExpr(const ComputeScalar scalar) {
  std::string expr = MetalType(scalar);
  expr += "(";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr +=
      scalar == ComputeScalar::Lane64 ? "0x8000000000000000ul" : "0x80000000u";
  expr += "))";
  return expr;
}

[[nodiscard]] inline std::string MetalFixedMaxExpr(const ComputeScalar scalar) {
  std::string expr = MetalType(scalar);
  expr += "(";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr +=
      scalar == ComputeScalar::Lane64 ? "0x7ffffffffffffffful" : "0x7fffffffu";
  expr += "))";
  return expr;
}

[[nodiscard]] inline std::string MetalWrapNegExpr(const ComputeScalar scalar,
                                                  const std::string &value) {
  std::string expr = MetalType(scalar);
  expr += "(";
  expr += MetalUnsignedType(scalar);
  expr += "(0) - ";
  expr += MetalUnsignedType(scalar);
  expr += "(";
  expr += value;
  expr += "))";
  return expr;
}

[[nodiscard]] inline std::string MetalAbsExpr(const ComputeScalar scalar,
                                              const std::string &value) {
  std::string expr = value;
  expr += " == ";
  expr += MetalFixedMinExpr(scalar);
  expr += " ? ";
  expr += MetalFixedMaxExpr(scalar);
  expr += " : (";
  expr += value;
  expr += " < ";
  expr += MetalZeroExpr(scalar);
  expr += " ? ";
  expr += MetalWrapNegExpr(scalar, value);
  expr += " : ";
  expr += value;
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
MetalAbsMagnitudeExpr(const ComputeScalar scalar, const std::string &value) {
  std::string expr = value;
  expr += " < ";
  expr += MetalZeroExpr(scalar);
  expr += " ? ";
  expr += MetalWrapNegExpr(scalar, value);
  expr += " : ";
  expr += value;
  return expr;
}

[[nodiscard]] inline std::string MetalSignExpr(const ComputeScalar scalar,
                                               const std::string &value) {
  std::string expr = value;
  expr += " > ";
  expr += MetalZeroExpr(scalar);
  expr += " ? ";
  expr += MetalOneExpr(scalar);
  expr += " : (";
  expr += value;
  expr += " < ";
  expr += MetalZeroExpr(scalar);
  expr += " ? ";
  expr += MetalMinusOneExpr(scalar);
  expr += " : ";
  expr += MetalZeroExpr(scalar);
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
MetalPredicateNotExpr(const ComputeScalar scalar, const std::string &value) {
  std::string expr = value;
  expr += " == ";
  expr += MetalZeroExpr(scalar);
  expr += " ? ";
  expr += MetalOneExpr(scalar);
  expr += " : ";
  expr += MetalZeroExpr(scalar);
  return expr;
}

[[nodiscard]] inline std::string
MetalPredicateLogicExpr(const ComputeScalar scalar, const std::string &lhs,
                        const char *const op_text, const std::string &rhs) {
  std::string expr = "(";
  expr += lhs;
  expr += " != ";
  expr += MetalZeroExpr(scalar);
  expr += ") ";
  expr += op_text;
  expr += " (";
  expr += rhs;
  expr += " != ";
  expr += MetalZeroExpr(scalar);
  expr += ") ? ";
  expr += MetalOneExpr(scalar);
  expr += " : ";
  expr += MetalZeroExpr(scalar);
  return expr;
}

[[nodiscard]] inline std::string
MetalSelectExpr(const ComputeScalar scalar, const std::string &condition,
                const std::string &when_true, const std::string &when_false) {
  const char *const scalar_type = MetalType(scalar);
  std::string expr = condition;
  expr += " != ";
  expr += scalar_type;
  expr += "(0) ? ";
  expr += when_true;
  expr += " : ";
  expr += when_false;
  return expr;
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
