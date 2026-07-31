#pragma once

#include <math64/prob/model.hpp>
#include <math64/soa/range.hpp>
#include <math64/soa/tail.hpp>

#include <array>
#include <bit>
#include <limits>

namespace rund::math64::prob::detail {

[[nodiscard]] inline RowStatus
ValidateRow(const soa::I64View input, const soa::I64MutView out) noexcept {
  if (input.size() != out.size()) return RowStatus{.processed = 0u, .size_match = false};
  if (soa::detail::PartiallyOverlaps(input, out)) return RowStatus{.processed = 0u, .overlap_ok = false};
  if (input.empty()) return RowStatus{.processed = 0u, .empty_input = true};
  return RowStatus{.processed = ::rund::math64::detail::ScalarSatU32(out.size())};
}
[[nodiscard]] constexpr i64 ClampWide(const ::rund::math64::detail::i128 value) noexcept {
  if (value > static_cast<::rund::math64::detail::i128>(FixedMax)) return FixedMax;
  if (value < static_cast<::rund::math64::detail::i128>(FixedMin)) return FixedMin;
  return static_cast<i64>(value);
}
[[nodiscard]] constexpr u64 BitLength128(::rund::math64::detail::u128 value) noexcept {
  u64 bits = 0u;
  while (value != 0u) { ++bits; value >>= 1u; }
  return bits;
}
[[nodiscard]] constexpr i64 ExpNegQ5_59ToQ1_63Approx(const i64 diff) noexcept {
  const auto base2 = (static_cast<::rund::math64::detail::i128>(diff) * static_cast<::rund::math64::detail::i128>(ProbLog2E_Q59)) /
                     static_cast<::rund::math64::detail::i128>(::rund::math64::detail::u128{1} << ProbLogitFractionBits);
  const auto magnitude = static_cast<::rund::math64::detail::u128>(-base2);
  const auto whole = magnitude >> ProbLogitFractionBits;
  const auto fraction = magnitude & ((::rund::math64::detail::u128{1} << ProbLogitFractionBits) - 1u);
  if (whole >= 63u) return 0;
  return ::rund::math64::detail::ScalarExp2(-static_cast<i64>(static_cast<u64>(fraction) << 4u)) >> static_cast<u64>(whole);
}
[[nodiscard]] inline simd::I64x ExpNegQ5_59ToQ1_63Approx(const simd::I64x diff) noexcept {
  const simd::I64x scaled = ::rund::math64::Clamp(diff << 4u, simd::SplatI64(FixedMin), simd::SplatI64(0));
  return ::rund::math64::Max(::rund::math64::Exp2(scaled), simd::SplatI64(0));
}
[[nodiscard]] inline ::rund::math64::detail::U128x2 AddSatWide(const ::rund::math64::detail::U128x2 lhs,
                                                               const ::rund::math64::detail::U128x2 rhs) noexcept {
  const ::rund::math64::detail::U128x2 sum = lhs + rhs;
  return ::rund::math64::detail::Select128(::rund::math64::detail::Gt128(lhs, sum),
                                          ::rund::math64::detail::SplatU128x2(~::rund::math64::detail::u128{0}),
                                          sum);
}
[[nodiscard]] inline ::rund::math64::detail::u128 ReduceAddSatWide(const ::rund::math64::detail::U128x2 value) noexcept {
  alignas(32) std::array<::rund::math64::detail::u128, simd::LaneCount> lanes{};
  __builtin_memcpy(lanes.data(), &value, sizeof(value));
  ::rund::math64::detail::u128 sum = 0u;
  for (const ::rund::math64::detail::u128 lane : lanes) {
    const ::rund::math64::detail::u128 next = sum + lane;
    sum = next < sum ? ~::rund::math64::detail::u128{0} : next;
  }
  return sum;
}
[[nodiscard]] inline simd::I64x NormalizeWeights(const simd::I64x weights,
                                                 const ::rund::math64::detail::u128 denominator) noexcept {
  const ::rund::math64::detail::U128x2 numerator =
      ::rund::math64::detail::Widen128(
          std::bit_cast<simd::U64x>(weights)) << 63u;
  const ::rund::math64::detail::U128x2 quotient =
      ::rund::math64::detail::UnsignedDiv128(numerator, ::rund::math64::detail::SplatU128x2(denominator));
  const ::rund::math64::detail::U128x2 clamped =
      ::rund::math64::detail::Select128(::rund::math64::detail::Gt128(quotient, ::rund::math64::detail::SplatU128x2(static_cast<u64>(FixedMax))),
                                        ::rund::math64::detail::SplatU128x2(static_cast<u64>(FixedMax)),
                                        quotient);
  return std::bit_cast<simd::I64x>(
      ::rund::math64::detail::Narrow64(clamped));
}
[[nodiscard]] inline simd::I64x ExpWeights(const soa::I64View logits,
                                           const std::size_t index,
                                           const simd::I64x max_value,
                                           const bool tail) noexcept {
  const simd::I64x loaded = tail ? soa::detail::LoadTailI64(logits, index, FixedMin)
                                 : simd::LoadI64(logits.data() + index);
  simd::I64x weights = ExpNegQ5_59ToQ1_63Approx(SubSat(loaded, max_value));
  if (tail) {
    weights = simd::Select(soa::detail::TailMask64(logits.size(), index), weights, simd::SplatI64(0));
  }
  return weights;
}
[[nodiscard]] inline ::rund::math64::detail::u128 WeightSumQ1_63(const soa::I64View logits, const simd::I64x max_value) noexcept {
  ::rund::math64::detail::U128x2 lane_sum = ::rund::math64::detail::SplatU128x2(0);
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= logits.size(); index += simd::LaneCount) {
    const simd::I64x weights = ExpWeights(logits, index, max_value, false);
    lane_sum = AddSatWide(
        lane_sum, ::rund::math64::detail::Widen128(
                      std::bit_cast<simd::U64x>(weights)));
  }
  if (index < logits.size()) {
    const simd::I64x weights = ExpWeights(logits, index, max_value, true);
    lane_sum = AddSatWide(
        lane_sum, ::rund::math64::detail::Widen128(
                      std::bit_cast<simd::U64x>(weights)));
  }
  return ReduceAddSatWide(lane_sum);
}
[[nodiscard]] constexpr ::rund::math64::detail::i128 LogWeightsQ5_59ApproxWide(const ::rund::math64::detail::u128 sum) noexcept {
  const u64 bits = BitLength128(sum);
  const auto mantissa = bits >= 63u ? (sum >> (bits - 63u)) : (sum << (63u - bits));
  const i64 log2_mantissa_q1_63 = ::rund::math64::detail::ScalarLog2(::rund::math64::detail::ScalarClamp(static_cast<::rund::math64::detail::i128>(mantissa)));
  const auto exponent_q5_59 = (static_cast<::rund::math64::detail::i128>(bits) - 63) *
                             static_cast<::rund::math64::detail::i128>(::rund::math64::detail::u128{1} << ProbLogitFractionBits);
  const auto log2_sum_q5_59 = exponent_q5_59 + static_cast<::rund::math64::detail::i128>(log2_mantissa_q1_63 / 16);
  return (log2_sum_q5_59 * static_cast<::rund::math64::detail::i128>(ProbLn2_Q59)) /
         static_cast<::rund::math64::detail::i128>(::rund::math64::detail::u128{1} << ProbLogitFractionBits);
}
}  // namespace rund::math64::prob::detail
