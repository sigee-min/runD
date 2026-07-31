#pragma once

#include <kernel/program/compute/lowering/names.hpp>
#include <kernel/program/compute/lowering/text.hpp>
#include <kernel/program/compute/lowering/vulkan/syntax.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] constexpr const char *
VulkanSignMask(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "0x8000000000000000ul"
                                          : "0x80000000u";
}

[[nodiscard]] inline std::string
VulkanSignedCompareExpr(const ComputeScalar scalar, const std::string &lhs,
                        const char *const op_text, const std::string &rhs) {
  const char *const mask = VulkanSignMask(scalar);
  std::string expr = "(";
  expr += lhs;
  expr += " ^ ";
  expr += mask;
  expr += ") ";
  expr += op_text;
  expr += " (";
  expr += rhs;
  expr += " ^ ";
  expr += mask;
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string VulkanCompareExpr(const ComputeDomain domain,
                                                   const ComputeScalar scalar,
                                                   const std::string &lhs,
                                                   const char *const op_text,
                                                   const std::string &rhs) {
  if (!UnsignedDomain(domain)) {
    return VulkanSignedCompareExpr(scalar, lhs, op_text, rhs);
  }
  return lhs + " " + op_text + " " + rhs;
}

[[nodiscard]] inline std::string VulkanHexLiteral(const ComputeScalar scalar,
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

[[nodiscard]] inline u64 VulkanHighMask(const ComputeScalar scalar,
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

[[nodiscard]] inline std::string
VulkanArithmeticShiftExpr(const ComputeScalar scalar, const std::string &value,
                          const u32 amount) {
  std::string expr = "(";
  expr += value;
  expr += " >> ";
  expr += std::to_string(amount);
  expr += "u) | ((";
  expr += value;
  expr += " & ";
  expr += VulkanHexLiteral(scalar, scalar == ComputeScalar::Lane64
                                       ? 0x8000000000000000ull
                                       : 0x80000000ull);
  expr += ") != ";
  expr += VulkanType(scalar);
  expr += "(0) ? ";
  expr += VulkanHexLiteral(scalar, VulkanHighMask(scalar, amount));
  expr += " : ";
  expr += VulkanType(scalar);
  expr += "(0))";
  return expr;
}

[[nodiscard]] inline std::string VulkanSignedMinExpr(const ComputeScalar scalar,
                                                     const std::string &lhs,
                                                     const std::string &rhs) {
  std::string expr = VulkanSignedCompareExpr(scalar, lhs, "<", rhs);
  expr += " ? ";
  expr += lhs;
  expr += " : ";
  expr += rhs;
  return expr;
}

[[nodiscard]] inline std::string VulkanSignedMaxExpr(const ComputeScalar scalar,
                                                     const std::string &lhs,
                                                     const std::string &rhs) {
  std::string expr = VulkanSignedCompareExpr(scalar, lhs, "<", rhs);
  expr += " ? ";
  expr += rhs;
  expr += " : ";
  expr += lhs;
  return expr;
}

[[nodiscard]] inline std::string VulkanMinExpr(const ComputeDomain domain,
                                               const ComputeScalar scalar,
                                               const std::string &lhs,
                                               const std::string &rhs) {
  std::string expr = VulkanCompareExpr(domain, scalar, lhs, "<", rhs);
  expr += " ? " + lhs + " : " + rhs;
  return expr;
}

[[nodiscard]] inline std::string VulkanMaxExpr(const ComputeDomain domain,
                                               const ComputeScalar scalar,
                                               const std::string &lhs,
                                               const std::string &rhs) {
  std::string expr = VulkanCompareExpr(domain, scalar, lhs, "<", rhs);
  expr += " ? " + rhs + " : " + lhs;
  return expr;
}

[[nodiscard]] inline std::string
VulkanPredicateExpr(const ComputeScalar scalar, const std::string &condition) {
  const char *const scalar_type = VulkanType(scalar);
  std::string expr = condition;
  expr += " ? ";
  expr += scalar_type;
  expr += "(1) : ";
  expr += scalar_type;
  expr += "(0)";
  return expr;
}

[[nodiscard]] inline std::string VulkanConstantExpr(const ComputeScalar scalar,
                                                    const u64 bits) {
  std::string expr = VulkanType(scalar);
  expr += "(";
  expr += std::to_string(bits);
  expr += scalar == ComputeScalar::Lane64 ? "ul" : "u";
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string VulkanZeroExpr(const ComputeScalar scalar) {
  std::string expr = VulkanType(scalar);
  expr += "(0)";
  return expr;
}

[[nodiscard]] inline std::string VulkanOneExpr(const ComputeScalar scalar) {
  std::string expr = VulkanType(scalar);
  expr += "(1)";
  return expr;
}

[[nodiscard]] inline std::string
VulkanMinusOneExpr(const ComputeScalar scalar) {
  return scalar == ComputeScalar::Lane64 ? "uint64_t(0xfffffffffffffffful)"
                                          : "uint(0xffffffffu)";
}

[[nodiscard]] inline std::string
VulkanFixedMinExpr(const ComputeScalar scalar) {
  return scalar == ComputeScalar::Lane64 ? "uint64_t(0x8000000000000000ul)"
                                          : "uint(0x80000000u)";
}

[[nodiscard]] inline std::string
VulkanFixedMaxExpr(const ComputeScalar scalar) {
  return scalar == ComputeScalar::Lane64 ? "uint64_t(0x7ffffffffffffffful)"
                                          : "uint(0x7fffffffu)";
}

[[nodiscard]] inline std::string VulkanWrapNegExpr(const ComputeScalar scalar,
                                                   const std::string &value) {
  std::string expr = VulkanType(scalar);
  expr += "(0) - ";
  expr += value;
  return expr;
}

[[nodiscard]] inline std::string VulkanAbsExpr(const ComputeScalar scalar,
                                               const std::string &value) {
  std::string expr = value;
  expr += " == ";
  expr += VulkanFixedMinExpr(scalar);
  expr += " ? ";
  expr += VulkanFixedMaxExpr(scalar);
  expr += " : (";
  expr += VulkanSignedCompareExpr(scalar, value, "<", VulkanZeroExpr(scalar));
  expr += " ? ";
  expr += VulkanWrapNegExpr(scalar, value);
  expr += " : ";
  expr += value;
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
VulkanAbsMagnitudeExpr(const ComputeScalar scalar, const std::string &value) {
  std::string expr =
      VulkanSignedCompareExpr(scalar, value, "<", VulkanZeroExpr(scalar));
  expr += " ? ";
  expr += VulkanWrapNegExpr(scalar, value);
  expr += " : ";
  expr += value;
  return expr;
}

[[nodiscard]] inline std::string VulkanSignExpr(const ComputeScalar scalar,
                                                const std::string &value) {
  std::string expr =
      VulkanSignedCompareExpr(scalar, value, ">", VulkanZeroExpr(scalar));
  expr += " ? ";
  expr += VulkanOneExpr(scalar);
  expr += " : (";
  expr += VulkanSignedCompareExpr(scalar, value, "<", VulkanZeroExpr(scalar));
  expr += " ? ";
  expr += VulkanMinusOneExpr(scalar);
  expr += " : ";
  expr += VulkanZeroExpr(scalar);
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
VulkanPredicateNotExpr(const ComputeScalar scalar, const std::string &value) {
  std::string expr = value;
  expr += " == ";
  expr += VulkanZeroExpr(scalar);
  expr += " ? ";
  expr += VulkanOneExpr(scalar);
  expr += " : ";
  expr += VulkanZeroExpr(scalar);
  return expr;
}

[[nodiscard]] inline std::string
VulkanPredicateLogicExpr(const ComputeScalar scalar, const std::string &lhs,
                         const char *const op_text, const std::string &rhs) {
  std::string expr = "(";
  expr += lhs;
  expr += " != ";
  expr += VulkanZeroExpr(scalar);
  expr += ") ";
  expr += op_text;
  expr += " (";
  expr += rhs;
  expr += " != ";
  expr += VulkanZeroExpr(scalar);
  expr += ") ? ";
  expr += VulkanOneExpr(scalar);
  expr += " : ";
  expr += VulkanZeroExpr(scalar);
  return expr;
}

[[nodiscard]] inline std::string
VulkanSelectExpr(const ComputeScalar scalar, const std::string &condition,
                 const std::string &when_true, const std::string &when_false) {
  const char *const scalar_type = VulkanType(scalar);
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
