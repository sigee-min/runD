#pragma once

#include <math64/turn/constants.hpp>

#include <bit>

namespace rund::math64 {
namespace detail {
[[nodiscard]] constexpr i64 TurnPair(const i64 even, const i64 odd, const i64 square) { return ::rund::math64::detail::ScalarAddSat(even, ::rund::math64::detail::ScalarMulFixed(odd, square)); }
[[nodiscard]] constexpr i64 TurnPoly10(const i64 square, const i64 (&coef)[10]) {
  i64 acc = coef[9];
  for (int i = 8; i >= 0; --i) acc = ::rund::math64::detail::ScalarAddSat(coef[i], ::rund::math64::detail::ScalarMulFixed(acc, square));
  return acc;
}
[[nodiscard]] constexpr i64 TurnSinUnit(const u64 offset) {
  if (offset == 0u) return 0;
  if (offset >= TurnOctant) return FixedSqrtHalf;
  const i64 unit = static_cast<i64>(offset << 2u);
  return ::rund::math64::detail::ScalarMulFixed(unit, TurnPoly10(::rund::math64::detail::ScalarMulFixed(unit, unit), TurnSinCoef));
}
[[nodiscard]] constexpr i64 TurnCosUnit(const u64 offset) {
  if (offset == 0u) return FixedMax;
  if (offset >= TurnOctant) return FixedSqrtHalf;
  const i64 unit = static_cast<i64>(offset << 2u);
  return TurnPoly10(::rund::math64::detail::ScalarMulFixed(unit, unit), TurnCosCoef);
}
[[nodiscard]] constexpr SinCos TurnFirstOctant(const u64 offset) { return SinCos{.sin = TurnSinUnit(offset), .cos = TurnCosUnit(offset)}; }
[[nodiscard]] inline simd::I64x TurnPair(const simd::I64x even, const simd::I64x odd, const simd::I64x square) noexcept {
  return AddSat(even, MulFixed(odd, square));
}
[[nodiscard]] inline simd::I64x TurnPoly10(const simd::I64x square, const i64 (&coef)[10]) noexcept {
  simd::I64x acc = simd::SplatI64(coef[9]);
  for (int i = 8; i >= 0; --i) acc = AddSat(simd::SplatI64(coef[i]), MulFixed(acc, square));
  return acc;
}
[[nodiscard]] inline simd::I64x TurnSinUnit(const simd::U64x offset) noexcept {
  const simd::I64x unit = std::bit_cast<simd::I64x>(offset << 2u);
  const simd::I64x approx = MulFixed(unit, TurnPoly10(MulFixed(unit, unit), TurnSinCoef));
  const simd::I64x with_zero = simd::Select(simd::Eq(offset, simd::SplatU64(0)), simd::SplatI64(0), approx);
  return simd::Select(simd::Ge(offset, simd::SplatU64(TurnOctant)), simd::SplatI64(FixedSqrtHalf), with_zero);
}
[[nodiscard]] inline simd::I64x TurnCosUnit(const simd::U64x offset) noexcept {
  const simd::I64x unit = std::bit_cast<simd::I64x>(offset << 2u);
  const simd::I64x approx = TurnPoly10(MulFixed(unit, unit), TurnCosCoef);
  const simd::I64x with_zero = simd::Select(simd::Eq(offset, simd::SplatU64(0)), simd::SplatI64(FixedMax), approx);
  return simd::Select(simd::Ge(offset, simd::SplatU64(TurnOctant)), simd::SplatI64(FixedSqrtHalf), with_zero);
}
[[nodiscard]] inline SinCosx TurnFirstOctant(const simd::U64x offset) noexcept {
  return SinCosx{.sin = TurnSinUnit(offset), .cos = TurnCosUnit(offset)};
}
}  // namespace detail

[[nodiscard]] constexpr SinCos TurnSinCos(const u64 turn) noexcept {
  const u64 octant = turn >> 61u;
  const u64 offset = turn & (TurnOctant - 1u);
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

[[nodiscard]] inline SinCosx TurnSinCos(const simd::U64x turn) noexcept {
  const simd::U64x octant = turn >> 61u;
  const simd::U64x offset = turn & simd::SplatU64(TurnOctant - 1u);
  const SinCosx a = detail::TurnFirstOctant(offset);
  const SinCosx b = detail::TurnFirstOctant(simd::SplatU64(TurnOctant) - offset);
  SinCosx out = a;
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU64(1)), b.cos, out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU64(1)), b.sin, out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU64(2)), a.cos, out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU64(2)), NegPositiveFixed(a.sin), out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU64(3)), b.sin, out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU64(3)), NegPositiveFixed(b.cos), out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU64(4)), NegPositiveFixed(a.sin), out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU64(4)), NegPositiveFixed(a.cos), out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU64(5)), NegPositiveFixed(b.cos), out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU64(5)), NegPositiveFixed(b.sin), out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU64(6)), NegPositiveFixed(a.cos), out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU64(6)), a.sin, out.cos)};
  out = SinCosx{.sin = simd::Select(simd::Eq(octant, simd::SplatU64(7)), NegPositiveFixed(b.sin), out.sin),
                .cos = simd::Select(simd::Eq(octant, simd::SplatU64(7)), b.cos, out.cos)};
  return out;
}
[[nodiscard]] inline simd::I64x TurnSin(const simd::U64x turn) noexcept { return TurnSinCos(turn).sin; }
[[nodiscard]] inline simd::I64x TurnCos(const simd::U64x turn) noexcept { return TurnSinCos(turn).cos; }
}  // namespace rund::math64
