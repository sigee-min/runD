#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/emission/reachability.hpp>
#include <kernel/program/compute/lowering/vulkan/nonlinear.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

template <typename Source>
inline void AppendVulkanFixedLane32Helpers(Source &out) {
  out += "uint RundAbsMagnitude32(uint value) {\n";
  out += "  return (value & 0x80000000u) == 0u ? value : (~value + 1u);\n";
  out += "}\n";
  out +=
      "uint RundClampSignedMagnitude32(bool negative, uint64_t magnitude) {\n";
  out += "  return negative ? (magnitude >= 0x80000000ul ? 0x80000000u : "
         "(0u - uint(magnitude))) : (magnitude > 0x7ffffffful ? 0x7fffffffu : "
         "uint(magnitude));\n";
  out += "}\n";
  out += "uint RundAddSat32(uint lhs, uint rhs) {\n";
  out += "  const uint sum = lhs + rhs;\n";
  out += "  const bool lhs_neg = (lhs & 0x80000000u) != 0u;\n";
  out += "  const bool rhs_neg = (rhs & 0x80000000u) != 0u;\n";
  out += "  const bool sum_neg = (sum & 0x80000000u) != 0u;\n";
  out += "  return (!lhs_neg && !rhs_neg && sum_neg) ? 0x7fffffffu : "
         "((lhs_neg && rhs_neg && !sum_neg) ? 0x80000000u : sum);\n";
  out += "}\n";
  out += "uint RundAddSatUnsigned32(uint lhs, uint rhs) {\n";
  out += "  const uint sum = lhs + rhs;\n";
  out += "  return sum < lhs ? 0xffffffffu : sum;\n";
  out += "}\n";
  out += "uint RundSubSat32(uint lhs, uint rhs) {\n";
  out += "  const uint diff = lhs - rhs;\n";
  out += "  const bool lhs_neg = (lhs & 0x80000000u) != 0u;\n";
  out += "  const bool rhs_neg = (rhs & 0x80000000u) != 0u;\n";
  out += "  const bool diff_neg = (diff & 0x80000000u) != 0u;\n";
  out += "  return (!lhs_neg && rhs_neg && diff_neg) ? 0x7fffffffu : "
         "((lhs_neg && !rhs_neg && !diff_neg) ? 0x80000000u : diff);\n";
  out += "}\n";
  out += "uint RundNegPositiveFixedLane32(uint value) {\n";
  out += "  return value == 0x7fffffffu ? 0x80000000u : (0u - value);\n";
  out += "}\n";
  out += "uint RundMulFixedLane32(uint lhs, uint rhs) {\n";
  out += "  const bool negative = ((lhs & 0x80000000u) != 0u) != "
         "((rhs & 0x80000000u) != 0u);\n";
  out += "  const uint64_t product = uint64_t(RundAbsMagnitude32(lhs)) * "
         "uint64_t(RundAbsMagnitude32(rhs));\n";
  out += "  return RundClampSignedMagnitude32(negative, uint(product >> "
         "31ul));\n";
  out += "}\n";
  out += "uint RundMulFixedScaled32(uint value, uint coefficient) {\n";
  out += "  const uint64_t product = uint64_t(RundAbsMagnitude32(value)) * "
         "uint64_t(coefficient);\n";
  out += "  return RundClampSignedMagnitude32((value & 0x80000000u) != 0u, "
         "uint(product >> 31ul));\n";
  out += "}\n";
  AppendVulkanFixedLane32NonlinearHelpers(out);
}

template <typename Source>
inline void AppendVulkanFixedLane64Helpers(Source &out) {
  out += "uint64_t RundAbsMagnitude64(uint64_t value) {\n";
  out += "  return (value & 0x8000000000000000ul) == 0ul ? value : "
         "(~value + 1ul);\n";
  out += "}\n";
  out += "uint64_t RundMulUnsignedShift63(uint64_t lhs, uint64_t rhs) {\n";
  out += "  const uint64_t mask32 = 0xfffffffful;\n";
  out += "  const uint64_t lhs_lo = lhs & mask32;\n";
  out += "  const uint64_t lhs_hi = lhs >> 32ul;\n";
  out += "  const uint64_t rhs_lo = rhs & mask32;\n";
  out += "  const uint64_t rhs_hi = rhs >> 32ul;\n";
  out += "  const uint64_t p0 = lhs_lo * rhs_lo;\n";
  out += "  const uint64_t p1 = lhs_lo * rhs_hi;\n";
  out += "  const uint64_t p2 = lhs_hi * rhs_lo;\n";
  out += "  const uint64_t p3 = lhs_hi * rhs_hi;\n";
  out += "  const uint64_t carry = (p0 >> 32ul) + (p1 & mask32) + "
         "(p2 & mask32);\n";
  out += "  const uint64_t high = p3 + (p1 >> 32ul) + (p2 >> 32ul) + "
         "(carry >> 32ul);\n";
  out += "  return (high << 1ul) | ((carry >> 31ul) & 1ul);\n";
  out += "}\n";
  out += "uint64_t RundClampSignedMagnitude64(bool negative, "
         "uint64_t magnitude) {\n";
  out += "  return negative ? (magnitude >= 0x8000000000000000ul ? "
         "0x8000000000000000ul : (0ul - magnitude)) : "
         "(magnitude > 0x7ffffffffffffffful ? 0x7ffffffffffffffful : "
         "magnitude);\n";
  out += "}\n";
  out += "uint64_t RundAddSat64(uint64_t lhs, uint64_t rhs) {\n";
  out += "  const uint64_t sum = lhs + rhs;\n";
  out += "  const bool lhs_neg = (lhs & 0x8000000000000000ul) != 0ul;\n";
  out += "  const bool rhs_neg = (rhs & 0x8000000000000000ul) != 0ul;\n";
  out += "  const bool sum_neg = (sum & 0x8000000000000000ul) != 0ul;\n";
  out += "  return (!lhs_neg && !rhs_neg && sum_neg) ? "
         "0x7ffffffffffffffful : ((lhs_neg && rhs_neg && !sum_neg) ? "
         "0x8000000000000000ul : sum);\n";
  out += "}\n";
  out += "uint64_t RundAddSatUnsigned64(uint64_t lhs, uint64_t rhs) {\n";
  out += "  const uint64_t sum = lhs + rhs;\n";
  out += "  return sum < lhs ? 0xfffffffffffffffful : sum;\n";
  out += "}\n";
  out += "uint64_t RundSubSat64(uint64_t lhs, uint64_t rhs) {\n";
  out += "  const uint64_t diff = lhs - rhs;\n";
  out += "  const bool lhs_neg = (lhs & 0x8000000000000000ul) != 0ul;\n";
  out += "  const bool rhs_neg = (rhs & 0x8000000000000000ul) != 0ul;\n";
  out += "  const bool diff_neg = (diff & 0x8000000000000000ul) != 0ul;\n";
  out += "  return (!lhs_neg && rhs_neg && diff_neg) ? "
         "0x7ffffffffffffffful : ((lhs_neg && !rhs_neg && !diff_neg) ? "
         "0x8000000000000000ul : diff);\n";
  out += "}\n";
  out += "uint64_t RundNegPositiveFixedLane64(uint64_t value) {\n";
  out += "  return value == 0x7ffffffffffffffful ? 0x8000000000000000ul : "
         "(0ul - value);\n";
  out += "}\n";
  out += "uint64_t RundMulFixedLane64(uint64_t lhs, uint64_t rhs) {\n";
  out += "  const bool negative = ((lhs & 0x8000000000000000ul) != 0ul) != "
         "((rhs & 0x8000000000000000ul) != 0ul);\n";
  out += "  return RundClampSignedMagnitude64(negative, "
         "RundMulUnsignedShift63(RundAbsMagnitude64(lhs), "
         "RundAbsMagnitude64(rhs)));\n";
  out += "}\n";
  out += "uint64_t RundMulFixedScaled64(uint64_t value, "
         "uint64_t coefficient) {\n";
  out += "  return RundClampSignedMagnitude64((value & "
         "0x8000000000000000ul) != 0ul, "
         "RundMulUnsignedShift63(RundAbsMagnitude64(value), coefficient));\n";
  out += "}\n";
  AppendVulkanFixedLane64NonlinearHelpers(out);
}

inline void AppendVulkanFixedOpHelpers(std::string &out, const ParsedIR &parsed,
                                       const ArtifactKey &key) {
  const std::vector<std::string> roots =
      CanonicalFixedHelperRoots(parsed, key.scalar);
  if (roots.empty()) {
    return;
  }
  std::string library;
  if (key.scalar == ComputeScalar::Lane64) {
    AppendVulkanFixedLane64Helpers(library);
  } else {
    AppendVulkanFixedLane32Helpers(library);
  }
  out += KeepReachableGeneratedHelpers(library, roots);
}

} // namespace compute_lowering_detail
} // namespace rund::kernel
