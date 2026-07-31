#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/emission/reachability.hpp>
#include <kernel/program/compute/lowering/metal/nonlinear.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

inline void AppendMetalFixedLane32Helpers(std::string &out) {
  out += "inline int RundClamp32(long value) {\n";
  out += "  return value > 2147483647l ? int(uint(0x7fffffffu)) : "
         "(value < (-2147483647l - 1l) ? int(uint(0x80000000u)) : "
         "int(value));\n";
  out += "}\n";
  out += "inline uint RundAbsMagnitude32(int value) {\n";
  out += "  const uint bits = uint(value);\n";
  out += "  return (bits & 0x80000000u) == 0u ? bits : (~bits + 1u);\n";
  out += "}\n";
  out += "inline int RundClampSignedMagnitude32(bool negative, "
         "ulong magnitude) {\n";
  out += "  return negative ? (magnitude >= 0x80000000ul ? "
         "int(uint(0x80000000u)) : int(uint(0u) - uint(magnitude))) : "
         "(magnitude > 0x7ffffffful ? int(uint(0x7fffffffu)) : "
         "int(uint(magnitude)));\n";
  out += "}\n";
  out += "inline int RundAddSat32(int lhs, int rhs) {\n";
  out += "  return RundClamp32(long(lhs) + long(rhs));\n";
  out += "}\n";
  out += "inline int RundAddSatUnsigned32(int lhs, int rhs) {\n";
  out += "  const uint lhs_bits = uint(lhs);\n";
  out += "  const uint sum = lhs_bits + uint(rhs);\n";
  out += "  return int(sum < lhs_bits ? 0xffffffffu : sum);\n";
  out += "}\n";
  out += "inline int RundSubSat32(int lhs, int rhs) {\n";
  out += "  return RundClamp32(long(lhs) - long(rhs));\n";
  out += "}\n";
  out += "inline int RundNegPositiveFixedLane32(int value) {\n";
  out += "  return value == int(uint(0x7fffffffu)) ? int(uint(0x80000000u)) "
         ": int(uint(0u) - uint(value));\n";
  out += "}\n";
  out += "inline int RundMulFixedLane32(int lhs, int rhs) {\n";
  out += "  const bool negative = (lhs < int(0)) != (rhs < int(0));\n";
  out += "  const ulong product = ulong(RundAbsMagnitude32(lhs)) * "
         "ulong(RundAbsMagnitude32(rhs));\n";
  out += "  return RundClampSignedMagnitude32(negative, uint(product >> "
         "31u));\n";
  out += "}\n";
  out += "inline int RundMulFixedScaled32(int value, int coefficient) {\n";
  out += "  const ulong product = ulong(RundAbsMagnitude32(value)) * "
         "ulong(uint(coefficient));\n";
  out += "  return RundClampSignedMagnitude32(value < int(0), uint(product >> "
         "31u));\n";
  out += "}\n";
  AppendMetalFixedLane32NonlinearHelpers(out);
}

inline void AppendMetalFixedLane64Helpers(std::string &out) {
  out += "inline long RundAsSigned64(ulong value) {\n";
  out += "  return as_type<long>(value);\n";
  out += "}\n";
  out += "inline ulong RundAsUnsigned64(long value) {\n";
  out += "  return as_type<ulong>(value);\n";
  out += "}\n";
  out += "inline ulong RundAbsMagnitude64(long value) {\n";
  out += "  const ulong bits = RundAsUnsigned64(value);\n";
  out += "  return (bits & 0x8000000000000000ul) == 0ul ? bits : "
         "(~bits + 1ul);\n";
  out += "}\n";
  out += "inline ulong RundMulUnsignedShift63(ulong lhs, ulong rhs) {\n";
  out += "  const ulong mask32 = 0xfffffffful;\n";
  out += "  const ulong lhs_lo = lhs & mask32;\n";
  out += "  const ulong lhs_hi = lhs >> 32u;\n";
  out += "  const ulong rhs_lo = rhs & mask32;\n";
  out += "  const ulong rhs_hi = rhs >> 32u;\n";
  out += "  const ulong p0 = lhs_lo * rhs_lo;\n";
  out += "  const ulong p1 = lhs_lo * rhs_hi;\n";
  out += "  const ulong p2 = lhs_hi * rhs_lo;\n";
  out += "  const ulong p3 = lhs_hi * rhs_hi;\n";
  out += "  const ulong carry = (p0 >> 32u) + (p1 & mask32) + "
         "(p2 & mask32);\n";
  out += "  const ulong high = p3 + (p1 >> 32u) + (p2 >> 32u) + "
         "(carry >> 32u);\n";
  out += "  return (high << 1u) | ((carry >> 31u) & 1ul);\n";
  out += "}\n";
  out += "inline long RundClampSignedMagnitude64(bool negative, "
         "ulong magnitude) {\n";
  out += "  if (negative) {\n";
  out += "    return magnitude >= 0x8000000000000000ul ? "
         "RundAsSigned64(0x8000000000000000ul) : "
         "RundAsSigned64(0ul - magnitude);\n";
  out += "  }\n";
  out += "  return magnitude > 0x7ffffffffffffffful ? "
         "RundAsSigned64(0x7ffffffffffffffful) : "
         "RundAsSigned64(magnitude);\n";
  out += "}\n";
  out += "inline long RundAddSat64(long lhs, long rhs) {\n";
  out += "  const ulong lhs_bits = RundAsUnsigned64(lhs);\n";
  out += "  const ulong rhs_bits = RundAsUnsigned64(rhs);\n";
  out += "  const ulong sum = lhs_bits + rhs_bits;\n";
  out += "  const bool lhs_neg = (lhs_bits & 0x8000000000000000ul) != 0ul;\n";
  out += "  const bool rhs_neg = (rhs_bits & 0x8000000000000000ul) != 0ul;\n";
  out += "  const bool sum_neg = (sum & 0x8000000000000000ul) != 0ul;\n";
  out += "  return (!lhs_neg && !rhs_neg && sum_neg) ? "
         "RundAsSigned64(0x7ffffffffffffffful) : ((lhs_neg && rhs_neg && "
         "!sum_neg) ? RundAsSigned64(0x8000000000000000ul) : "
         "RundAsSigned64(sum));\n";
  out += "}\n";
  out += "inline long RundAddSatUnsigned64(long lhs, long rhs) {\n";
  out += "  const ulong lhs_bits = RundAsUnsigned64(lhs);\n";
  out += "  const ulong sum = lhs_bits + RundAsUnsigned64(rhs);\n";
  out += "  return RundAsSigned64(sum < lhs_bits ? 0xfffffffffffffffful : "
         "sum);\n";
  out += "}\n";
  out += "inline long RundSubSat64(long lhs, long rhs) {\n";
  out += "  const ulong lhs_bits = RundAsUnsigned64(lhs);\n";
  out += "  const ulong rhs_bits = RundAsUnsigned64(rhs);\n";
  out += "  const ulong diff = lhs_bits - rhs_bits;\n";
  out += "  const bool lhs_neg = (lhs_bits & 0x8000000000000000ul) != 0ul;\n";
  out += "  const bool rhs_neg = (rhs_bits & 0x8000000000000000ul) != 0ul;\n";
  out += "  const bool diff_neg = (diff & 0x8000000000000000ul) != 0ul;\n";
  out += "  return (!lhs_neg && rhs_neg && diff_neg) ? "
         "RundAsSigned64(0x7ffffffffffffffful) : ((lhs_neg && !rhs_neg && "
         "!diff_neg) ? RundAsSigned64(0x8000000000000000ul) : "
         "RundAsSigned64(diff));\n";
  out += "}\n";
  out += "inline long RundNegPositiveFixedLane64(long value) {\n";
  out += "  return value == RundAsSigned64(0x7ffffffffffffffful) ? "
         "RundAsSigned64(0x8000000000000000ul) : "
         "RundAsSigned64(0ul - RundAsUnsigned64(value));\n";
  out += "}\n";
  out += "inline long RundMulFixedLane64(long lhs, long rhs) {\n";
  out += "  const bool negative = (lhs < long(0)) != (rhs < long(0));\n";
  out += "  return RundClampSignedMagnitude64(negative, "
         "RundMulUnsignedShift63(RundAbsMagnitude64(lhs), "
         "RundAbsMagnitude64(rhs)));\n";
  out += "}\n";
  out += "inline long RundMulFixedScaled64(long value, long coefficient) {\n";
  out += "  return RundClampSignedMagnitude64(value < long(0), "
         "RundMulUnsignedShift63(RundAbsMagnitude64(value), "
         "RundAsUnsigned64(coefficient)));\n";
  out += "}\n";
  AppendMetalFixedLane64NonlinearHelpers(out);
}

inline void AppendMetalFixedOpHelpers(std::string &out, const ParsedIR &parsed,
                                      const ArtifactKey &key) {
  const std::vector<std::string> roots =
      CanonicalFixedHelperRoots(parsed, key.scalar);
  if (roots.empty()) {
    return;
  }
  std::string library;
  if (key.scalar == ComputeScalar::Lane64) {
    AppendMetalFixedLane64Helpers(library);
  } else {
    AppendMetalFixedLane32Helpers(library);
  }
  out += KeepReachableGeneratedHelpers(library, roots);
}

} // namespace compute_lowering_detail
} // namespace rund::kernel
