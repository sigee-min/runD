#pragma once

#include <math32/turn/constants.hpp>

#include <bit>

namespace rund::math32 {
namespace detail {
[[nodiscard]] constexpr i32 TurnPair(const i32 even, const i32 odd, const i32 square) {
  return ::rund::math32::detail::ScalarAddSat(even, ::rund::math32::detail::ScalarMulFixed(odd, square));
}
[[nodiscard]] constexpr i32 TurnPoly6(const i32 square, const i32 (&coef)[6]) {
  const i32 square2 = ::rund::math32::detail::ScalarMulFixed(square, square);
  const i32 square4 = ::rund::math32::detail::ScalarMulFixed(square2, square2);
  const i32 p0 = TurnPair(coef[0], coef[1], square);
  const i32 p1 = TurnPair(coef[2], coef[3], square);
  const i32 p2 = TurnPair(coef[4], coef[5], square);
  return ::rund::math32::detail::ScalarAddSat(::rund::math32::detail::ScalarAddSat(p0, ::rund::math32::detail::ScalarMulFixed(p1, square2)), ::rund::math32::detail::ScalarMulFixed(p2, square4));
}
[[nodiscard]] constexpr i32 TurnSinUnit(const u32 offset) {
  if (offset == 0u) return 0;
  if (offset >= TurnOctant) return FixedSqrtHalf;
  const i32 unit = static_cast<i32>(offset << 2u);
  return ::rund::math32::detail::ScalarMulFixed(unit, TurnPoly6(::rund::math32::detail::ScalarMulFixed(unit, unit), TurnSinCoef));
}
[[nodiscard]] constexpr i32 TurnCosUnit(const u32 offset) {
  if (offset == 0u) return FixedMax;
  if (offset >= TurnOctant) return FixedSqrtHalf;
  const i32 unit = static_cast<i32>(offset << 2u);
  return TurnPoly6(::rund::math32::detail::ScalarMulFixed(unit, unit), TurnCosCoef);
}
[[nodiscard]] constexpr SinCos TurnFirstOctant(const u32 offset) {
  return SinCos{.sin = TurnSinUnit(offset), .cos = TurnCosUnit(offset)};
}
[[nodiscard]] inline simd::I32x TurnPair(const simd::I32x even, const simd::I32x odd, const simd::I32x square) noexcept {
  return AddSat(even, MulFixed(odd, square));
}
[[nodiscard]] inline simd::I32x TurnPoly6(const simd::I32x square, const i32 (&coef)[6]) noexcept {
  const simd::I32x square2 = MulFixed(square, square);
  const simd::I32x square4 = MulFixed(square2, square2);
  const simd::I32x p0 = TurnPair(simd::SplatI32(coef[0]), simd::SplatI32(coef[1]), square);
  const simd::I32x p1 = TurnPair(simd::SplatI32(coef[2]), simd::SplatI32(coef[3]), square);
  const simd::I32x p2 = TurnPair(simd::SplatI32(coef[4]), simd::SplatI32(coef[5]), square);
  return AddSat(AddSat(p0, MulFixed(p1, square2)), MulFixed(p2, square4));
}
[[nodiscard]] inline simd::I32x TurnSinUnit(const simd::U32x offset) noexcept {
  const simd::I32x unit = std::bit_cast<simd::I32x>(offset << 2u);
  const simd::I32x approx = MulFixed(unit, TurnPoly6(MulFixed(unit, unit), TurnSinCoef));
  const simd::I32x with_zero = simd::Select(simd::Eq(offset, simd::SplatU32(0)), simd::SplatI32(0), approx);
  return simd::Select(simd::Ge(offset, simd::SplatU32(TurnOctant)), simd::SplatI32(FixedSqrtHalf), with_zero);
}
[[nodiscard]] inline simd::I32x TurnCosUnit(const simd::U32x offset) noexcept {
  const simd::I32x unit = std::bit_cast<simd::I32x>(offset << 2u);
  const simd::I32x approx = TurnPoly6(MulFixed(unit, unit), TurnCosCoef);
  const simd::I32x with_zero = simd::Select(simd::Eq(offset, simd::SplatU32(0)), simd::SplatI32(FixedMax), approx);
  return simd::Select(simd::Ge(offset, simd::SplatU32(TurnOctant)), simd::SplatI32(FixedSqrtHalf), with_zero);
}
[[nodiscard]] inline SinCosx TurnFirstOctant(const simd::U32x offset) noexcept {
  return SinCosx{.sin = TurnSinUnit(offset), .cos = TurnCosUnit(offset)};
}
}  // namespace detail

[[nodiscard]] constexpr SinCos TurnSinCos(const u32 turn) noexcept {
  const u32 octant = turn >> 29u;
  const u32 offset = turn & (TurnOctant - 1u);
  const SinCos a = detail::TurnFirstOctant(offset);
  const SinCos b = detail::TurnFirstOctant(TurnOctant - offset);
  switch (octant) {
  case 1u: return SinCos{.sin = b.cos, .cos = b.sin};
  case 2u: return SinCos{.sin = a.cos, .cos = detail::ScalarNegPositiveFixed(a.sin)};
  case 3u: return SinCos{.sin = b.sin, .cos = detail::ScalarNegPositiveFixed(b.cos)};
  case 4u: return SinCos{.sin = detail::ScalarNegPositiveFixed(a.sin), .cos = detail::ScalarNegPositiveFixed(a.cos)};
  case 5u: return SinCos{.sin = detail::ScalarNegPositiveFixed(b.cos), .cos = detail::ScalarNegPositiveFixed(b.sin)};
  case 6u: return SinCos{.sin = detail::ScalarNegPositiveFixed(a.cos), .cos = a.sin};
  case 7u: return SinCos{.sin = detail::ScalarNegPositiveFixed(b.sin), .cos = b.cos};
  default: return a;
  }
}

[[nodiscard]] inline SinCosx TurnSinCos(const simd::U32x turn) noexcept {
  const simd::U32x octant = turn >> 29u;
  const simd::U32x offset = turn & simd::SplatU32(TurnOctant - 1u);
  const SinCosx a = detail::TurnFirstOctant(offset);
  const SinCosx b = detail::TurnFirstOctant(simd::SplatU32(TurnOctant) - offset);
  SinCosx out = a;
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU32(1)), b.cos, out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU32(1)), b.sin, out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU32(2)), a.cos, out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU32(2)), NegPositiveFixed(a.sin), out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU32(3)), b.sin, out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU32(3)), NegPositiveFixed(b.cos), out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU32(4)), NegPositiveFixed(a.sin), out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU32(4)), NegPositiveFixed(a.cos), out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU32(5)), NegPositiveFixed(b.cos), out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU32(5)), NegPositiveFixed(b.sin), out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU32(6)), NegPositiveFixed(a.cos), out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU32(6)), a.sin, out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU32(7)), NegPositiveFixed(b.sin), out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU32(7)), b.cos, out.cos)};
  return out;
}
[[nodiscard]] inline simd::I32x TurnSin(const simd::U32x turn) noexcept { return TurnSinCos(turn).sin; }
[[nodiscard]] inline simd::I32x TurnCos(const simd::U32x turn) noexcept { return TurnSinCos(turn).cos; }
}  // namespace rund::math32
