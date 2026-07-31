#pragma once

#include <kernel/program/compute/lowering/fixed/64/turn.hpp>
#include <kernel/program/compute/lowering/vulkan/u128.hpp>

#include <string>

namespace rund::kernel {
namespace compute_lowering_detail {

template <std::size_t N>
inline void AppendVulkanTurnPoly64(
    std::string &out, const char *const name,
    const std::array<std::int64_t, N> &coefficients) {
  out += "uint64_t ";
  out += name;
  out += "(uint64_t square) {\n";
  out += "  uint64_t acc = ";
  out += std::to_string(static_cast<std::uint64_t>(coefficients.back()));
  out += "ul;\n";
  for (std::size_t index = N - 1u; index != 0u; --index) {
    out += "  acc = RundAddSat64(";
    out += std::to_string(
        static_cast<std::uint64_t>(coefficients[index - 1u]));
    out += "ul, RundMulFixedLane64(acc, square));\n";
  }
  out += "  return acc;\n}\n";
}

inline void AppendVulkanFixedLane32NonlinearHelpers(std::string &out) {
  out += "uint64_t RundTurnUnsignedDiv32(uint64_t numerator, "
         "uint denominator) {\n";
  out += "  uint64_t quotient = 0ul;\n";
  out += "  uint64_t remainder = 0ul;\n";
  out += "  for (int bit = 62; bit >= 0; --bit) {\n";
  out += "    remainder = (remainder << 1ul) | ((numerator >> "
         "uint64_t(bit)) & 1ul);\n";
  out += "    if (remainder >= uint64_t(denominator)) {\n";
  out += "      remainder -= uint64_t(denominator);\n";
  out += "      quotient |= 1ul << uint64_t(bit);\n";
  out += "    }\n";
  out += "  }\n";
  out += "  return quotient;\n";
  out += "}\n";
  out += "uint RundTurnPair32(uint even, uint odd, uint square) {\n";
  out += "  return RundAddSat32(even, RundMulFixedLane32(odd, square));\n";
  out += "}\n";
  out += "uint RundTurnPoly6_32(uint square, bool cosine) {\n";
  out += "  const uint square2 = RundMulFixedLane32(square, square);\n";
  out += "  const uint square4 = RundMulFixedLane32(square2, square2);\n";
  out += "  const uint p0 = RundTurnPair32(cosine ? 2147483647u : 1686629713u, cosine ? uint(-662337939) : uint(-173399667), square);\n";
  out += "  const uint p1 = RundTurnPair32(cosine ? 34046945u : 5348082u, cosine ? uint(-700062) : uint(-78547), square);\n";
  out += "  const uint p2 = RundTurnPair32(cosine ? 7711u : 673u, cosine ? uint(-53) : uint(-4), square);\n";
  out += "  return RundAddSat32(RundAddSat32(p0, RundMulFixedLane32(p1, square2)), RundMulFixedLane32(p2, square4));\n";
  out += "}\n";
  out += "uint RundTurnUnit32(uint offset, bool cosine) {\n";
  out += "  if (offset == 0u) { return cosine ? 0x7fffffffu : 0u; }\n";
  out += "  if (offset >= 0x20000000u) { return 1518500250u; }\n";
  out += "  const uint unit = offset << 2u;\n";
  out += "  const uint square = RundMulFixedLane32(unit, unit);\n";
  out += "  const uint poly = RundTurnPoly6_32(square, cosine);\n";
  out += "  return cosine ? poly : RundMulFixedLane32(unit, poly);\n";
  out += "}\n";
  out += "uint RundTurnTrig32(uint turn, bool cosine) {\n";
  out += "  const uint octant = turn >> 29u;\n";
  out += "  const uint offset = turn & 0x1fffffffu;\n";
  out += "  const uint as = RundTurnUnit32(offset, false);\n";
  out += "  const uint ac = RundTurnUnit32(offset, true);\n";
  out += "  const uint bs = RundTurnUnit32(0x20000000u - offset, false);\n";
  out += "  const uint bc = RundTurnUnit32(0x20000000u - offset, true);\n";
  out += "  uint s = as; uint c = ac;\n";
  out += "  if (octant == 1u) { s = bc; c = bs; }\n";
  out += "  else if (octant == 2u) { s = ac; c = RundNegPositiveFixedLane32(as); }\n";
  out += "  else if (octant == 3u) { s = bs; c = RundNegPositiveFixedLane32(bc); }\n";
  out += "  else if (octant == 4u) { s = RundNegPositiveFixedLane32(as); c = RundNegPositiveFixedLane32(ac); }\n";
  out += "  else if (octant == 5u) { s = RundNegPositiveFixedLane32(bc); c = RundNegPositiveFixedLane32(bs); }\n";
  out += "  else if (octant == 6u) { s = RundNegPositiveFixedLane32(ac); c = as; }\n";
  out += "  else if (octant == 7u) { s = RundNegPositiveFixedLane32(bs); c = bc; }\n";
  out += "  return cosine ? c : s;\n";
  out += "}\n";
  out += "uint RundSin32(uint value) { return RundTurnTrig32(value, false); }\n";
  out += "uint RundCos32(uint value) { return RundTurnTrig32(value, true); }\n";
  out += "uint RundExp32(uint value) {\n";
  out += "  if ((value & 0x80000000u) == 0u) { return 0x7fffffffu; }\n";
  out += "  if (value == 0x80000000u) { return 0x40000000u; }\n";
  out += "  const uint unit = RundAbsMagnitude32(value);\n";
  out += "  uint polynomial = uint(330788);\n";
  out += "  polynomial = RundAddSat32(uint(-2863360), RundMulFixedLane32(polynomial, unit));\n";
  out += "  polynomial = RundAddSat32(uint(20654775), RundMulFixedLane32(polynomial, unit));\n";
  out += "  polynomial = RundAddSat32(uint(-119194166), RundMulFixedLane32(polynomial, unit));\n";
  out += "  polynomial = RundAddSat32(uint(515882496), RundMulFixedLane32(polynomial, unit));\n";
  out += "  polynomial = RundAddSat32(uint(-1488522236), RundMulFixedLane32(polynomial, unit));\n";
  out += "  const uint approximated = RundAddSat32(0x7fffffffu, "
         "RundMulFixedLane32(polynomial, unit));\n";
  out += "  return approximated < 0x40000000u ? 0x40000000u : approximated;\n";
  out += "}\n";
  out += "uint RundLog32(uint value) {\n";
  out += "  if ((value & 0x80000000u) != 0u || value <= 0x40000000u) { return "
         "0x80000000u; }\n";
  out += "  if (value >= 0x7fffffffu) { return 0u; }\n";
  out += "  const uint offset = RundSubSat32(value, 0x7fffffffu);\n";
  out += "  uint power = offset;\n";
  out += "  uint result = RundMulFixedScaled32(power, 3098164009u);\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundSubSat32(result, RundMulFixedScaled32(power, 1549082005u));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundAddSat32(result, RundMulFixedScaled32(power, 1032721336u));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundSubSat32(result, RundMulFixedScaled32(power, 774541002u));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundAddSat32(result, RundMulFixedScaled32(power, 619632802u));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundSubSat32(result, RundMulFixedScaled32(power, 516360668u));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  return RundAddSat32(result, RundMulFixedScaled32(power, 442594858u));\n";
  out += "}\n";
  out += "uint RundTurnPoly12_32(uint square) {\n";
  out += "  const uint s2 = RundMulFixedLane32(square, square);\n";
  out += "  const uint s4 = RundMulFixedLane32(s2, s2);\n";
  out += "  const uint s8 = RundMulFixedLane32(s4, s4);\n";
  out += "  const uint p0 = RundTurnPair32(1367130551u, uint(-455708559), square);\n";
  out += "  const uint p1 = RundTurnPair32(273424749u, uint(-195303391), square);\n";
  out += "  const uint p2 = RundTurnPair32(151887082u, uint(-124270335), square);\n";
  out += "  const uint p3 = RundTurnPair32(105459068u, uint(-91860930), square);\n";
  out += "  const uint p4 = RundTurnPair32(81654160u, uint(-73704920), square);\n";
  out += "  const uint p5 = RundTurnPair32(67352940u, uint(-62155800), square);\n";
  out += "  return RundAddSat32(RundAddSat32(RundAddSat32(p0, RundMulFixedLane32(p1, s2)), RundMulFixedLane32(RundAddSat32(p2, RundMulFixedLane32(p3, s2)), s4)), RundMulFixedLane32(RundAddSat32(p4, RundMulFixedLane32(p5, s2)), s8));\n";
  out += "}\n";
  out += "uint RundTurnRatio32(uint numerator, uint denominator) {\n";
  out += "  if (denominator == 0u) { return 0x7fffffffu; }\n";
  out += "  return uint(min(RundTurnUnsignedDiv32(uint64_t(numerator) << 31ul, denominator), uint64_t(0x7fffffffu)));\n";
  out += "}\n";
  out += "uint RundTurnAtanUnit32(uint ratio) {\n";
  out += "  return RundMulFixedLane32(ratio, RundTurnPoly12_32(RundMulFixedLane32(ratio, ratio)));\n";
  out += "}\n";
  out += "uint RundTurnAtanRatio32(uint small, uint large) {\n";
  out += "  if (small == 0u || large == 0u) { return 0u; }\n";
  out += "  const uint direct = RundTurnAtanUnit32(RundTurnRatio32(small, large));\n";
  out += "  const uint delta_ratio = RundTurnRatio32(large - small, large + small);\n";
  out += "  const uint delta = RundAddSat32(0x20000000u, RundTurnAtanUnit32(delta_ratio));\n";
  out += "  const bool direct_range = uint64_t(small) * 0x80000000ul <= uint64_t(large) * 889516852ul;\n";
  out += "  return direct_range ? direct : delta;\n";
  out += "}\n";
  out += "uint RundAtan232(uint y, uint x) {\n";
  out += "  const uint ya = RundAbsMagnitude32(y);\n";
  out += "  const uint xa = RundAbsMagnitude32(x);\n";
  out += "  if (xa == 0u && ya == 0u) { return 0u; }\n";
  out += "  const bool y_small = ya <= xa;\n";
  out += "  const uint offset0 = RundTurnAtanRatio32(y_small ? ya : xa, y_small ? xa : ya);\n";
  out += "  const uint offset = y_small ? offset0 : 0x40000000u - offset0;\n";
  out += "  uint turn = 0u - offset;\n";
  out += "  const bool x_neg = (x & 0x80000000u) != 0u;\n";
  out += "  const bool y_neg = (y & 0x80000000u) != 0u;\n";
  out += "  if (x_neg && y_neg) { turn = 0x80000000u + offset; }\n";
  out += "  else if (x_neg && !y_neg && y != 0u) { turn = 0x80000000u - offset; }\n";
  out += "  else if (!x_neg && x != 0u && !y_neg && y != 0u) { turn = offset; }\n";
  out += "  if (y == 0u) { turn = x_neg ? 0x80000000u : 0u; }\n";
  out += "  if (x == 0u) { turn = !y_neg && y != 0u ? 0x40000000u : 0xc0000000u; }\n";
  out += "  return turn;\n";
  out += "}\n";
}

inline void AppendVulkanFixedLane64NonlinearHelpers(std::string &out) {
  AppendVulkanU128Core(out);
  AppendVulkanU128Division(out);
  AppendVulkanU128IntegerSquareRoot(out);
  out += "RundU128 RundTurnNumerator64(uint64_t magnitude) {\n";
  out += "  return RundMakeU128(magnitude >> 1ul, magnitude << 63ul);\n";
  out += "}\n";
  AppendVulkanTurnPoly64(out, "RundTurnSinPoly64", kTurnSin64);
  AppendVulkanTurnPoly64(out, "RundTurnCosPoly64", kTurnCos64);
  AppendVulkanTurnPoly64(out, "RundTurnAtanPoly64", kTurnAtan64);
  out += "uint64_t RundTurnUnit64(uint64_t offset, bool cosine) {\n";
  out += "  if (offset == 0ul) { return cosine ? 0x7ffffffffffffffful : 0ul; }\n";
  out += "  if (offset >= 0x2000000000000000ul) { return ";
  out += std::to_string(kTurnSqrtHalf64);
  out += "ul; }\n";
  out += "  const uint64_t unit = offset << 2ul;\n";
  out += "  const uint64_t square = RundMulFixedLane64(unit, unit);\n";
  out += "  const uint64_t polynomial = cosine ? RundTurnCosPoly64(square) : RundTurnSinPoly64(square);\n";
  out += "  return cosine ? polynomial : RundMulFixedLane64(unit, polynomial);\n";
  out += "}\n";
  out += "uint64_t RundTurnTrig64(uint64_t turn, bool cosine) {\n";
  out += "  const uint64_t octant = turn >> 61ul;\n";
  out += "  const uint64_t offset = turn & 0x1ffffffffffffffful;\n";
  out += "  const uint64_t as = RundTurnUnit64(offset, false);\n";
  out += "  const uint64_t ac = RundTurnUnit64(offset, true);\n";
  out += "  const uint64_t bs = RundTurnUnit64(0x2000000000000000ul - offset, false);\n";
  out += "  const uint64_t bc = RundTurnUnit64(0x2000000000000000ul - offset, true);\n";
  out += "  uint64_t sine = as;\n";
  out += "  uint64_t cosine_value = ac;\n";
  out += "  if (octant == 1ul) { sine = bc; cosine_value = bs; }\n";
  out += "  if (octant == 2ul) { sine = ac; cosine_value = RundNegPositiveFixedLane64(as); }\n";
  out += "  if (octant == 3ul) { sine = bs; cosine_value = RundNegPositiveFixedLane64(bc); }\n";
  out += "  if (octant == 4ul) { sine = RundNegPositiveFixedLane64(as); cosine_value = RundNegPositiveFixedLane64(ac); }\n";
  out += "  if (octant == 5ul) { sine = RundNegPositiveFixedLane64(bc); cosine_value = RundNegPositiveFixedLane64(bs); }\n";
  out += "  if (octant == 6ul) { sine = RundNegPositiveFixedLane64(ac); cosine_value = as; }\n";
  out += "  if (octant == 7ul) { sine = RundNegPositiveFixedLane64(bs); cosine_value = bc; }\n";
  out += "  return cosine ? cosine_value : sine;\n";
  out += "}\n";
  out += "uint64_t RundSin64(uint64_t value) { return RundTurnTrig64(value, false); }\n";
  out += "uint64_t RundCos64(uint64_t value) { return RundTurnTrig64(value, true); }\n";
  out += "uint64_t RundExp64(uint64_t value) {\n";
  out += "  if ((value & 0x8000000000000000ul) == 0ul) { return "
         "0x7ffffffffffffffful; }\n";
  out += "  if (value == 0x8000000000000000ul) { return "
         "0x4000000000000000ul; }\n";
  out += "  const uint64_t unit = RundAbsMagnitude64(value);\n";
  out += "  uint64_t polynomial = uint64_t(1420724914991586ul);\n";
  out += "  polynomial = RundAddSat64(uint64_t(-12298036735954530l), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  polynomial = RundAddSat64(uint64_t(88711583058159475ul), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  polynomial = RundAddSat64(uint64_t(-511935043789664227l), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  polynomial = RundAddSat64(uint64_t(2215698446797868712ul), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  polynomial = RundAddSat64(uint64_t(-6393154322601327830l), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  const uint64_t approximated = RundAddSat64(0x7ffffffffffffffful, "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  return approximated < 0x4000000000000000ul ? "
         "0x4000000000000000ul : approximated;\n";
  out += "}\n";
  out += "uint64_t RundLog64(uint64_t value) {\n";
  out += "  if ((value & 0x8000000000000000ul) != 0ul || value <= "
         "0x4000000000000000ul) { return 0x8000000000000000ul; }\n";
  out += "  if (value >= 0x7ffffffffffffffful) { return 0ul; }\n";
  out += "  const uint64_t offset = RundSubSat64(value, 0x7ffffffffffffffful);\n";
  out += "  uint64_t power = offset;\n";
  out += "  uint64_t result = RundMulFixedScaled64(power, 13306513097844322492ul);\n";
  out += "  power = RundMulFixedLane64(power, offset);\n";
  out += "  result = RundSubSat64(result, RundMulFixedScaled64(power, "
         "6653256548922161246ul));\n";
  out += "  power = RundMulFixedLane64(power, offset);\n";
  out += "  return RundAddSat64(result, RundMulFixedScaled64(power, "
         "4435504365948107497ul));\n";
  out += "}\n";
  out += "uint64_t RundTurnRatio64(uint64_t numerator, uint64_t denominator) {\n";
  out += "  if (denominator == 0ul) { return 0x7ffffffffffffffful; }\n";
  out += "  const uint64_t ratio = RundUnsignedDivU128ByU64(RundTurnNumerator64(numerator), denominator);\n";
  out += "  return ratio > 0x7ffffffffffffffful ? 0x7ffffffffffffffful : ratio;\n";
  out += "}\n";
  out += "bool RundTurnRatioLe64(uint64_t numerator, uint64_t denominator) {\n";
  out += "  if (denominator == 0ul) { return false; }\n";
  out += "  const RundU128 left = RundTurnNumerator64(numerator);\n";
  out += "  const RundU128 right = RundMulWide64(denominator, ";
  out += std::to_string(kTurnTanEighth64);
  out += "ul);\n";
  out += "  return RundGeU128(right, left);\n";
  out += "}\n";
  out += "uint64_t RundTurnAtanUnit64(uint64_t ratio) {\n";
  out += "  return RundMulFixedLane64(ratio, RundTurnAtanPoly64(RundMulFixedLane64(ratio, ratio)));\n";
  out += "}\n";
  out += "uint64_t RundTurnAtanRatio64(uint64_t small, uint64_t large) {\n";
  out += "  if (small == 0ul || large == 0ul) { return 0ul; }\n";
  out += "  const uint64_t direct = RundTurnAtanUnit64(RundTurnRatio64(small, large));\n";
  out += "  const uint64_t delta_ratio = RundTurnRatio64(large - small, large + small);\n";
  out += "  const uint64_t delta = RundAddSat64(0x2000000000000000ul, RundTurnAtanUnit64(delta_ratio));\n";
  out += "  return RundTurnRatioLe64(small, large) ? direct : delta;\n";
  out += "}\n";
  out += "uint64_t RundAtan264(uint64_t y, uint64_t x) {\n";
  out += "  const uint64_t ya = RundAbsMagnitude64(y);\n";
  out += "  const uint64_t xa = RundAbsMagnitude64(x);\n";
  out += "  if (xa == 0ul && ya == 0ul) { return 0ul; }\n";
  out += "  const bool y_small = ya <= xa;\n";
  out += "  const uint64_t offset0 = RundTurnAtanRatio64(y_small ? ya : xa, y_small ? xa : ya);\n";
  out += "  const uint64_t offset = y_small ? offset0 : 0x4000000000000000ul - offset0;\n";
  out += "  const bool x_neg = (x & 0x8000000000000000ul) != 0ul;\n";
  out += "  const bool y_neg = (y & 0x8000000000000000ul) != 0ul;\n";
  out += "  uint64_t turn = 0ul - offset;\n";
  out += "  if (x_neg && y_neg) { turn = 0x8000000000000000ul + offset; }\n";
  out += "  if (x_neg && !y_neg && y != 0ul) { turn = 0x8000000000000000ul - offset; }\n";
  out += "  if (!x_neg && x != 0ul && !y_neg && y != 0ul) { turn = offset; }\n";
  out += "  if (y == 0ul) { turn = x_neg ? 0x8000000000000000ul : 0ul; }\n";
  out += "  if (x == 0ul) { turn = !y_neg && y != 0ul ? 0x4000000000000000ul : 0xc000000000000000ul; }\n";
  out += "  return turn;\n";
  out += "}\n";
}

} // namespace compute_lowering_detail
} // namespace rund::kernel
