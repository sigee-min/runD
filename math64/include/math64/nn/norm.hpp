#pragma once

#include <math64/nn/activation.hpp>
#include <math64/soa/range.hpp>
#include <math64/soa/tail.hpp>

#include <array>
#include <limits>

namespace rund::math64::nn {
namespace detail {
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
[[nodiscard]] inline i64 RowMeanSquare(const soa::I64View input) noexcept {
  ::rund::math64::detail::U128x2 lane_sum = ::rund::math64::detail::SplatU128x2(0);
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= input.size(); index += simd::LaneCount) {
    const simd::U64x magnitude = AbsMagnitude(simd::LoadI64(input.data() + index));
    lane_sum = AddSatWide(
        lane_sum, ::rund::math64::detail::Widen128(
                      MulUnsignedFixed(magnitude, magnitude)));
  }
  if (index < input.size()) {
    const simd::U64x magnitude = AbsMagnitude(soa::detail::LoadTailI64(input, index));
    lane_sum = AddSatWide(
        lane_sum, ::rund::math64::detail::Widen128(
                      MulUnsignedFixed(magnitude, magnitude)));
  }
  const ::rund::math64::detail::u128 mean_square =
      input.empty() ? 0u : ReduceAddSatWide(lane_sum) / static_cast<::rund::math64::detail::u128>(input.size());
  return mean_square > static_cast<::rund::math64::detail::u128>(static_cast<u64>(FixedMax))
             ? FixedMax
             : static_cast<i64>(static_cast<u64>(mean_square));
}
}  // namespace detail

[[nodiscard]] inline RowStatus RmsNorm(const soa::I64View input,
                                       const soa::I64View weight,
                                       const soa::I64MutView out,
                                       const simd::I64x epsilon) noexcept {
  if (input.size() != weight.size() || input.size() != out.size()) return RowStatus{.processed = 0u, .size_match = false};
  if (input.empty()) return RowStatus{.processed = 0u, .empty_input = true};
  if (simd::Any(simd::Lt(epsilon, simd::SplatI64(0)))) return RowStatus{.processed = 0u, .valid_epsilon = false};
  if (soa::detail::PartiallyOverlaps(input, out) ||
      soa::detail::PartiallyOverlaps(weight, out)) {
    return RowStatus{.processed = 0u, .overlap_ok = false};
  }
  const simd::I64x denominator = Sqrt(AddSat(simd::SplatI64(detail::RowMeanSquare(input)), epsilon));
  u64 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= out.size(); index += simd::LaneCount) {
    const simd::I64x value = simd::LoadI64(input.data() + index);
    const simd::I64x scale = simd::LoadI64(weight.data() + index);
    simd::Store(out.data() + index, DivFixed(MulFixed(value, scale), denominator));
    processed += static_cast<u64>(simd::LaneCount);
  }
  if (index < out.size()) {
    const simd::I64x value = soa::detail::LoadTailI64(input, index);
    const simd::I64x scale = soa::detail::LoadTailI64(weight, index);
    soa::detail::StoreTailI64(out, index, DivFixed(MulFixed(value, scale), denominator));
    processed += ::rund::math64::detail::ScalarSatU32(out.size() - index);
  }
  return RowStatus{.processed = processed};
}
}  // namespace rund::math64::nn
