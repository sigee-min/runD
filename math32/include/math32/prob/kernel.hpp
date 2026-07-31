#pragma once

#include <math32/prob/model.hpp>
#include <math32/soa/range.hpp>
#include <math32/soa/tail.hpp>

#include <array>
#include <bit>
#include <limits>

namespace rund::math32::prob::detail {

[[nodiscard]] inline RowStatus
ValidateRow(const soa::I32View input, const soa::I32MutView out) noexcept {
  if (input.size() != out.size()) return RowStatus{.processed = 0u, .size_match = false};
  if (soa::detail::PartiallyOverlaps(input, out)) return RowStatus{.processed = 0u, .overlap_ok = false};
  if (input.empty()) return RowStatus{.processed = 0u, .empty_input = true};
  return RowStatus{.processed = ::rund::math32::detail::ScalarSatU32(out.size())};
}
[[nodiscard]] constexpr i32 ClampWide(const ::rund::math32::detail::i128 value) noexcept {
  if (value > static_cast<::rund::math32::detail::i128>(FixedMax)) return FixedMax;
  if (value < static_cast<::rund::math32::detail::i128>(FixedMin)) return FixedMin;
  return static_cast<i32>(value);
}
[[nodiscard]] constexpr u32 BitLength128(::rund::math32::detail::u128 value) noexcept {
  u32 bits = 0u;
  while (value != 0u) { ++bits; value >>= 1u; }
  return bits;
}
[[nodiscard]] constexpr i32 ExpNegQ5_27ToQ1_31Approx(const i64 diff) noexcept {
  const auto base2 = (static_cast<::rund::math32::detail::i128>(diff) * static_cast<::rund::math32::detail::i128>(ProbLog2E_Q27)) / static_cast<::rund::math32::detail::i128>(i64{1} << ProbLogitFractionBits);
  const auto magnitude = static_cast<::rund::math32::detail::u128>(-base2);
  const auto whole = magnitude >> ProbLogitFractionBits;
  const auto fraction = magnitude & ((::rund::math32::detail::u128{1} << ProbLogitFractionBits) - 1u);
  if (whole >= 31u) return 0;
  return ::rund::math32::detail::ScalarExp2(-static_cast<i32>(static_cast<u32>(fraction) << 4u)) >> static_cast<u32>(whole);
}
[[nodiscard]] inline simd::I32x ExpNegQ5_27ToQ1_31Approx(const simd::I32x diff) noexcept {
  const simd::I32x scaled = ::rund::math32::Clamp(diff << 4u, simd::SplatI32(FixedMin), simd::SplatI32(0));
  return ::rund::math32::Max(::rund::math32::Exp2(scaled), simd::SplatI32(0));
}
[[nodiscard]] inline ::rund::math32::detail::U64x4 AddSatWide(const ::rund::math32::detail::U64x4 lhs,
                                                              const ::rund::math32::detail::U64x4 rhs) noexcept {
  const ::rund::math32::detail::U64x4 sum = lhs + rhs;
  return ::rund::math32::detail::Select64(::rund::math32::detail::Gt64(lhs, sum),
                                          ::rund::math32::detail::SplatU64x4(std::numeric_limits<u64>::max()),
                                          sum);
}
[[nodiscard]] inline u64 ReduceAddSatWide(const ::rund::math32::detail::U64x4 value) noexcept {
  alignas(32) std::array<u64, simd::LaneCount> lanes{};
  __builtin_memcpy(lanes.data(), &value, sizeof(value));
  u64 sum = 0u;
  for (const u64 lane : lanes) {
    sum = ::rund::math32::detail::ScalarSatAdd(sum, lane);
  }
  return sum;
}
[[nodiscard]] inline simd::I32x NormalizeWeights(const simd::I32x weights, const u64 denominator) noexcept {
  const ::rund::math32::detail::U64x4 numerator =
      ::rund::math32::detail::Widen64(
          std::bit_cast<simd::U32x>(weights)) << 31u;
  return ::rund::math32::detail::ClampU64x4ToI32x(
      ::rund::math32::detail::UnsignedDiv64(numerator, ::rund::math32::detail::SplatU64x4(denominator)));
}
[[nodiscard]] inline simd::I32x ExpWeights(const soa::I32View logits,
                                           const std::size_t index,
                                           const simd::I32x max_value,
                                           const bool tail) noexcept {
  const simd::I32x loaded = tail ? soa::detail::LoadTailI32(logits, index, FixedMin)
                                 : simd::LoadI32(logits.data() + index);
  simd::I32x weights = ExpNegQ5_27ToQ1_31Approx(SubSat(loaded, max_value));
  if (tail) {
    weights = simd::Select(soa::detail::TailMask32(logits.size(), index), weights, simd::SplatI32(0));
  }
  return weights;
}
[[nodiscard]] inline u64 WeightSumQ1_31(const soa::I32View logits, const simd::I32x max_value) noexcept {
  ::rund::math32::detail::U64x4 lane_sum = ::rund::math32::detail::SplatU64x4(0);
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= logits.size(); index += simd::LaneCount) {
    const simd::I32x weights = ExpWeights(logits, index, max_value, false);
    lane_sum = AddSatWide(
        lane_sum, ::rund::math32::detail::Widen64(
                      std::bit_cast<simd::U32x>(weights)));
  }
  if (index < logits.size()) {
    const simd::I32x weights = ExpWeights(logits, index, max_value, true);
    lane_sum = AddSatWide(
        lane_sum, ::rund::math32::detail::Widen64(
                      std::bit_cast<simd::U32x>(weights)));
  }
  return ReduceAddSatWide(lane_sum);
}
[[nodiscard]] constexpr ::rund::math32::detail::i128 LogWeightsQ5_27ApproxWide(const ::rund::math32::detail::u128 sum) noexcept {
  const u32 bits = BitLength128(sum);
  const auto mantissa = bits >= 31u ? (sum >> (bits - 31u)) : (sum << (31u - bits));
  const i32 log2_mantissa_q1_31 = ::rund::math32::detail::ScalarLog2(::rund::math32::detail::ScalarClamp(static_cast<i64>(mantissa)));
  const auto exponent_q5_27 = (static_cast<::rund::math32::detail::i128>(bits) - 31) * static_cast<::rund::math32::detail::i128>(i64{1} << ProbLogitFractionBits);
  const auto log2_sum_q5_27 = exponent_q5_27 + static_cast<::rund::math32::detail::i128>(log2_mantissa_q1_31 / 16);
  return (log2_sum_q5_27 * static_cast<::rund::math32::detail::i128>(ProbLn2_Q27)) / static_cast<::rund::math32::detail::i128>(i64{1} << ProbLogitFractionBits);
}
}  // namespace rund::math32::prob::detail
