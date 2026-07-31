#pragma once

#include <kernel/program/compute/lowering/fixed/64/turn.hpp>
#include <kernel/program/compute/lowering/metal/u128.hpp>

#include <string>

namespace rund::kernel {
namespace compute_lowering_detail {

template <std::size_t N>
inline void AppendMetalTurnPoly64(
    std::string &out, const char *const name,
    const std::array<std::int64_t, N> &coefficients) {
  out += "inline long ";
  out += name;
  out += "(long square) {\n";
  out += "  long acc = RundAsSigned64(";
  out += std::to_string(static_cast<std::uint64_t>(coefficients.back()));
  out += "ul);\n";
  for (std::size_t index = N - 1u; index != 0u; --index) {
    out += "  acc = RundAddSat64(RundAsSigned64(";
    out += std::to_string(
        static_cast<std::uint64_t>(coefficients[index - 1u]));
    out += "ul), RundMulFixedLane64(acc, square));\n";
  }
  out += "  return acc;\n}\n";
}

inline void AppendMetalFixedLane32NonlinearHelpers(std::string &out) {
  out += "inline ulong RundTurnUnsignedDiv32(ulong numerator, "
         "uint denominator) {\n";
  out += "  ulong quotient = 0ul;\n";
  out += "  ulong remainder = 0ul;\n";
  out += "  for (int bit = 62; bit >= 0; --bit) {\n";
  out += "    remainder = (remainder << 1u) | ((numerator >> uint(bit)) & "
         "1ul);\n";
  out += "    if (remainder >= ulong(denominator)) {\n";
  out += "      remainder -= ulong(denominator);\n";
  out += "      quotient |= 1ul << uint(bit);\n";
  out += "    }\n";
  out += "  }\n";
  out += "  return quotient;\n";
  out += "}\n";
  out += "inline int RundTurnPair32(int even, int odd, int square) {\n";
  out += "  return RundAddSat32(even, RundMulFixedLane32(odd, square));\n";
  out += "}\n";
  out += "inline int RundTurnPoly6_32(int square, bool cosine) {\n";
  out += "  const int square2 = RundMulFixedLane32(square, square);\n";
  out += "  const int square4 = RundMulFixedLane32(square2, square2);\n";
  out += "  const int p0 = RundTurnPair32(cosine ? 2147483647 : 1686629713, cosine ? -662337939 : -173399667, square);\n";
  out += "  const int p1 = RundTurnPair32(cosine ? 34046945 : 5348082, cosine ? -700062 : -78547, square);\n";
  out += "  const int p2 = RundTurnPair32(cosine ? 7711 : 673, cosine ? -53 : -4, square);\n";
  out += "  return RundAddSat32(RundAddSat32(p0, RundMulFixedLane32(p1, square2)), RundMulFixedLane32(p2, square4));\n";
  out += "}\n";
  out += "inline int RundTurnUnit32(uint offset, bool cosine) {\n";
  out += "  if (offset == 0u) { return cosine ? int(0x7fffffffu) : int(0); }\n";
  out += "  if (offset >= 0x20000000u) { return int(1518500250); }\n";
  out += "  const int unit = int(offset << 2u);\n";
  out += "  const int square = RundMulFixedLane32(unit, unit);\n";
  out += "  const int poly = RundTurnPoly6_32(square, cosine);\n";
  out += "  return cosine ? poly : RundMulFixedLane32(unit, poly);\n";
  out += "}\n";
  out += "inline int RundTurnTrig32(uint turn, bool cosine) {\n";
  out += "  const uint octant = turn >> 29u;\n";
  out += "  const uint offset = turn & 0x1fffffffu;\n";
  out += "  const int as = RundTurnUnit32(offset, false);\n";
  out += "  const int ac = RundTurnUnit32(offset, true);\n";
  out += "  const int bs = RundTurnUnit32(0x20000000u - offset, false);\n";
  out += "  const int bc = RundTurnUnit32(0x20000000u - offset, true);\n";
  out += "  int s = as; int c = ac;\n";
  out += "  if (octant == 1u) { s = bc; c = bs; }\n";
  out += "  else if (octant == 2u) { s = ac; c = RundNegPositiveFixedLane32(as); }\n";
  out += "  else if (octant == 3u) { s = bs; c = RundNegPositiveFixedLane32(bc); }\n";
  out += "  else if (octant == 4u) { s = RundNegPositiveFixedLane32(as); c = RundNegPositiveFixedLane32(ac); }\n";
  out += "  else if (octant == 5u) { s = RundNegPositiveFixedLane32(bc); c = RundNegPositiveFixedLane32(bs); }\n";
  out += "  else if (octant == 6u) { s = RundNegPositiveFixedLane32(ac); c = as; }\n";
  out += "  else if (octant == 7u) { s = RundNegPositiveFixedLane32(bs); c = bc; }\n";
  out += "  return cosine ? c : s;\n";
  out += "}\n";
  out += "inline int RundSin32(int value) { return RundTurnTrig32(uint(value), false); }\n";
  out += "inline int RundCos32(int value) { return RundTurnTrig32(uint(value), true); }\n";
  out += "inline int RundExp32(int value) {\n";
  out += "  if (value >= int(0)) { return int(uint(0x7fffffffu)); }\n";
  out += "  if (uint(value) == 0x80000000u) { return int(uint(0x40000000u)); "
         "}\n";
  out += "  const int unit = int(RundAbsMagnitude32(value));\n";
  out += "  int polynomial = int(330788);\n";
  out += "  polynomial = RundAddSat32(int(-2863360), "
         "RundMulFixedLane32(polynomial, unit));\n";
  out += "  polynomial = RundAddSat32(int(20654775), "
         "RundMulFixedLane32(polynomial, unit));\n";
  out += "  polynomial = RundAddSat32(int(-119194166), "
         "RundMulFixedLane32(polynomial, unit));\n";
  out += "  polynomial = RundAddSat32(int(515882496), "
         "RundMulFixedLane32(polynomial, unit));\n";
  out += "  polynomial = RundAddSat32(int(-1488522236), "
         "RundMulFixedLane32(polynomial, unit));\n";
  out += "  return min(max(RundAddSat32(int(uint(0x7fffffffu)), "
         "RundMulFixedLane32(polynomial, unit)), int(uint(0x40000000u))), "
         "int(uint(0x7fffffffu)));\n";
  out += "}\n";
  out += "inline int RundLog32(int value) {\n";
  out += "  if (value <= int(uint(0x40000000u))) { return int(uint(0x80000000u)); }\n";
  out += "  if (value >= int(uint(0x7fffffffu))) { return int(0); }\n";
  out += "  const int offset = RundSubSat32(value, int(uint(0x7fffffffu)));\n";
  out += "  int power = offset;\n";
  out += "  int result = RundMulFixedScaled32(power, int(uint(3098164009u)));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundSubSat32(result, "
         "RundMulFixedScaled32(power, int(uint(1549082005u))));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundAddSat32(result, "
         "RundMulFixedScaled32(power, int(uint(1032721336u))));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundSubSat32(result, "
         "RundMulFixedScaled32(power, int(uint(774541002u))));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundAddSat32(result, "
         "RundMulFixedScaled32(power, int(uint(619632802u))));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundSubSat32(result, "
         "RundMulFixedScaled32(power, int(uint(516360668u))));\n";
  out += "  power = RundMulFixedLane32(power, offset);\n";
  out += "  result = RundAddSat32(result, "
         "RundMulFixedScaled32(power, int(uint(442594858u))));\n";
  out += "  return min(max(result, int(uint(0x80000000u))), int(0));\n";
  out += "}\n";
  out += "inline int RundTurnPoly12_32(int square) {\n";
  out += "  const int s2 = RundMulFixedLane32(square, square);\n";
  out += "  const int s4 = RundMulFixedLane32(s2, s2);\n";
  out += "  const int s8 = RundMulFixedLane32(s4, s4);\n";
  out += "  const int p0 = RundTurnPair32(1367130551, -455708559, square);\n";
  out += "  const int p1 = RundTurnPair32(273424749, -195303391, square);\n";
  out += "  const int p2 = RundTurnPair32(151887082, -124270335, square);\n";
  out += "  const int p3 = RundTurnPair32(105459068, -91860930, square);\n";
  out += "  const int p4 = RundTurnPair32(81654160, -73704920, square);\n";
  out += "  const int p5 = RundTurnPair32(67352940, -62155800, square);\n";
  out += "  return RundAddSat32(RundAddSat32(RundAddSat32(p0, RundMulFixedLane32(p1, s2)), RundMulFixedLane32(RundAddSat32(p2, RundMulFixedLane32(p3, s2)), s4)), RundMulFixedLane32(RundAddSat32(p4, RundMulFixedLane32(p5, s2)), s8));\n";
  out += "}\n";
  out += "inline uint RundTurnRatio32(uint numerator, uint denominator) {\n";
  out += "  if (denominator == 0u) { return 0x7fffffffu; }\n";
  out += "  return uint(min(RundTurnUnsignedDiv32(ulong(numerator) << 31u, denominator), ulong(0x7fffffffu)));\n";
  out += "}\n";
  out += "inline int RundTurnAtanUnit32(uint ratio) {\n";
  out += "  const int r = int(ratio);\n";
  out += "  return RundMulFixedLane32(r, RundTurnPoly12_32(RundMulFixedLane32(r, r)));\n";
  out += "}\n";
  out += "inline uint RundTurnAtanRatio32(uint small, uint large) {\n";
  out += "  if (small == 0u || large == 0u) { return 0u; }\n";
  out += "  const uint direct = uint(RundTurnAtanUnit32(RundTurnRatio32(small, large)));\n";
  out += "  const uint delta_ratio = RundTurnRatio32(large - small, large + small);\n";
  out += "  const uint delta = uint(RundAddSat32(int(0x20000000u), RundTurnAtanUnit32(delta_ratio)));\n";
  out += "  const bool direct_range = ulong(small) * 0x80000000ul <= ulong(large) * 889516852ul;\n";
  out += "  return direct_range ? direct : delta;\n";
  out += "}\n";
  out += "inline int RundAtan232(int y, int x) {\n";
  out += "  const uint ya = RundAbsMagnitude32(y);\n";
  out += "  const uint xa = RundAbsMagnitude32(x);\n";
  out += "  if (xa == 0u && ya == 0u) { return int(0); }\n";
  out += "  const bool y_small = ya <= xa;\n";
  out += "  const uint offset0 = RundTurnAtanRatio32(y_small ? ya : xa, y_small ? xa : ya);\n";
  out += "  const uint offset = y_small ? offset0 : 0x40000000u - offset0;\n";
  out += "  uint turn = 0u - offset;\n";
  out += "  if (x < 0 && y < 0) { turn = 0x80000000u + offset; }\n";
  out += "  else if (x < 0 && y > 0) { turn = 0x80000000u - offset; }\n";
  out += "  else if (x > 0 && y > 0) { turn = offset; }\n";
  out += "  if (y == 0) { turn = x < 0 ? 0x80000000u : 0u; }\n";
  out += "  if (x == 0) { turn = y > 0 ? 0x40000000u : 0xc0000000u; }\n";
  out += "  return int(turn);\n";
  out += "}\n";
}

inline void AppendMetalFixedLane64NonlinearHelpers(std::string &out) {
  AppendMetalU128Core(out);
  AppendMetalU128Division(out);
  AppendMetalU128IntegerSquareRoot(out);
  out += "inline RundU128 RundTurnNumerator64(ulong magnitude) {\n";
  out += "  return RundMakeU128(magnitude >> 1u, magnitude << 63u);\n";
  out += "}\n";
  AppendMetalTurnPoly64(out, "RundTurnSinPoly64", kTurnSin64);
  AppendMetalTurnPoly64(out, "RundTurnCosPoly64", kTurnCos64);
  AppendMetalTurnPoly64(out, "RundTurnAtanPoly64", kTurnAtan64);
  out += "inline long RundTurnUnit64(ulong offset, bool cosine) {\n";
  out += "  if (offset == 0ul) { return cosine ? "
         "RundAsSigned64(0x7ffffffffffffffful) : long(0); }\n";
  out += "  if (offset >= 0x2000000000000000ul) { return RundAsSigned64(";
  out += std::to_string(kTurnSqrtHalf64);
  out += "ul); }\n";
  out += "  const long unit = RundAsSigned64(offset << 2u);\n";
  out += "  const long square = RundMulFixedLane64(unit, unit);\n";
  out += "  const long polynomial = cosine ? RundTurnCosPoly64(square) : "
         "RundTurnSinPoly64(square);\n";
  out += "  return cosine ? polynomial : RundMulFixedLane64(unit, polynomial);\n";
  out += "}\n";
  out += "inline long RundTurnTrig64(ulong turn, bool cosine) {\n";
  out += "  const ulong octant = turn >> 61u;\n";
  out += "  const ulong offset = turn & 0x1ffffffffffffffful;\n";
  out += "  const long as = RundTurnUnit64(offset, false);\n";
  out += "  const long ac = RundTurnUnit64(offset, true);\n";
  out += "  const long bs = RundTurnUnit64(0x2000000000000000ul - offset, false);\n";
  out += "  const long bc = RundTurnUnit64(0x2000000000000000ul - offset, true);\n";
  out += "  long sine = as;\n";
  out += "  long cosine_value = ac;\n";
  out += "  if (octant == 1ul) { sine = bc; cosine_value = bs; }\n";
  out += "  if (octant == 2ul) { sine = ac; cosine_value = RundNegPositiveFixedLane64(as); }\n";
  out += "  if (octant == 3ul) { sine = bs; cosine_value = RundNegPositiveFixedLane64(bc); }\n";
  out += "  if (octant == 4ul) { sine = RundNegPositiveFixedLane64(as); cosine_value = RundNegPositiveFixedLane64(ac); }\n";
  out += "  if (octant == 5ul) { sine = RundNegPositiveFixedLane64(bc); cosine_value = RundNegPositiveFixedLane64(bs); }\n";
  out += "  if (octant == 6ul) { sine = RundNegPositiveFixedLane64(ac); cosine_value = as; }\n";
  out += "  if (octant == 7ul) { sine = RundNegPositiveFixedLane64(bs); cosine_value = bc; }\n";
  out += "  return cosine ? cosine_value : sine;\n";
  out += "}\n";
  out += "inline long RundSin64(long value) { return RundTurnTrig64(RundAsUnsigned64(value), false); }\n";
  out += "inline long RundCos64(long value) { return RundTurnTrig64(RundAsUnsigned64(value), true); }\n";
  out += "inline long RundExp64(long value) {\n";
  out += "  if (value >= long(0)) { return RundAsSigned64(0x7ffffffffffffffful); "
         "}\n";
  out += "  if (RundAsUnsigned64(value) == 0x8000000000000000ul) { return "
         "RundAsSigned64(0x4000000000000000ul); }\n";
  out += "  const long unit = RundAsSigned64(RundAbsMagnitude64(value));\n";
  out += "  long polynomial = long(1420724914991586l);\n";
  out += "  polynomial = RundAddSat64(long(-12298036735954530l), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  polynomial = RundAddSat64(long(88711583058159475l), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  polynomial = RundAddSat64(long(-511935043789664227l), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  polynomial = RundAddSat64(long(2215698446797868712l), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  polynomial = RundAddSat64(long(-6393154322601327830l), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  const long approximated = RundAddSat64("
         "RundAsSigned64(0x7ffffffffffffffful), "
         "RundMulFixedLane64(polynomial, unit));\n";
  out += "  return approximated < RundAsSigned64(0x4000000000000000ul) ? "
         "RundAsSigned64(0x4000000000000000ul) : approximated;\n";
  out += "}\n";
  out += "inline long RundLog64(long value) {\n";
  out += "  if (value <= RundAsSigned64(0x4000000000000000ul)) { return "
         "RundAsSigned64(0x8000000000000000ul); }\n";
  out += "  if (value >= RundAsSigned64(0x7ffffffffffffffful)) { return long(0); }\n";
  out += "  const long offset = RundSubSat64(value, "
         "RundAsSigned64(0x7ffffffffffffffful));\n";
  out += "  long power = offset;\n";
  out += "  long result = RundMulFixedScaled64(power, "
         "RundAsSigned64(13306513097844322492ul));\n";
  out += "  power = RundMulFixedLane64(power, offset);\n";
  out += "  result = RundSubSat64(result, RundMulFixedScaled64(power, "
         "RundAsSigned64(6653256548922161246ul)));\n";
  out += "  power = RundMulFixedLane64(power, offset);\n";
  out += "  result = RundAddSat64(result, RundMulFixedScaled64(power, "
         "RundAsSigned64(4435504365948107497ul)));\n";
  out += "  return result > long(0) ? long(0) : result;\n";
  out += "}\n";
  out += "inline ulong RundTurnRatio64(ulong numerator, ulong denominator) {\n";
  out += "  if (denominator == 0ul) { return 0x7ffffffffffffffful; }\n";
  out += "  const ulong ratio = RundUnsignedDivU128ByU64(RundTurnNumerator64(numerator), denominator);\n";
  out += "  return ratio > 0x7ffffffffffffffful ? 0x7ffffffffffffffful : ratio;\n";
  out += "}\n";
  out += "inline bool RundTurnRatioLe64(ulong numerator, ulong denominator) {\n";
  out += "  if (denominator == 0ul) { return false; }\n";
  out += "  const RundU128 left = RundTurnNumerator64(numerator);\n";
  out += "  const RundU128 right = RundMulWide64(denominator, ";
  out += std::to_string(kTurnTanEighth64);
  out += "ul);\n";
  out += "  return RundGeU128(right, left);\n";
  out += "}\n";
  out += "inline long RundTurnAtanUnit64(ulong ratio) {\n";
  out += "  const long value = RundAsSigned64(ratio);\n";
  out += "  return RundMulFixedLane64(value, RundTurnAtanPoly64(RundMulFixedLane64(value, value)));\n";
  out += "}\n";
  out += "inline ulong RundTurnAtanRatio64(ulong small, ulong large) {\n";
  out += "  if (small == 0ul || large == 0ul) { return 0ul; }\n";
  out += "  const ulong direct = RundAsUnsigned64(RundTurnAtanUnit64(RundTurnRatio64(small, large)));\n";
  out += "  const ulong delta_ratio = RundTurnRatio64(large - small, large + small);\n";
  out += "  const ulong delta = RundAsUnsigned64(RundAddSat64(RundAsSigned64(0x2000000000000000ul), RundTurnAtanUnit64(delta_ratio)));\n";
  out += "  return RundTurnRatioLe64(small, large) ? direct : delta;\n";
  out += "}\n";
  out += "inline long RundAtan264(long y, long x) {\n";
  out += "  const ulong ya = RundAbsMagnitude64(y);\n";
  out += "  const ulong xa = RundAbsMagnitude64(x);\n";
  out += "  if (xa == 0ul && ya == 0ul) { return long(0); }\n";
  out += "  const bool y_small = ya <= xa;\n";
  out += "  const ulong offset0 = RundTurnAtanRatio64(y_small ? ya : xa, y_small ? xa : ya);\n";
  out += "  const ulong offset = y_small ? offset0 : 0x4000000000000000ul - offset0;\n";
  out += "  ulong turn = 0ul - offset;\n";
  out += "  if (x < long(0) && y < long(0)) { turn = 0x8000000000000000ul + offset; }\n";
  out += "  if (x < long(0) && y > long(0)) { turn = 0x8000000000000000ul - offset; }\n";
  out += "  if (x > long(0) && y > long(0)) { turn = offset; }\n";
  out += "  if (y == long(0)) { turn = x < long(0) ? 0x8000000000000000ul : 0ul; }\n";
  out += "  if (x == long(0)) { turn = y > long(0) ? 0x4000000000000000ul : 0xc000000000000000ul; }\n";
  out += "  return RundAsSigned64(turn);\n";
  out += "}\n";
}

} // namespace compute_lowering_detail
} // namespace rund::kernel
