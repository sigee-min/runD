#pragma once

#include <math32/nn/activation.hpp>
#include <math32/soa/range.hpp>
#include <math32/soa/tail.hpp>

#include <array>
#include <limits>

namespace rund::math32::nn {
namespace detail {
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
[[nodiscard]] inline i32 RowMeanSquare(const soa::I32View input) noexcept {
  ::rund::math32::detail::U64x4 lane_sum = ::rund::math32::detail::SplatU64x4(0);
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= input.size(); index += simd::LaneCount) {
    const simd::U32x magnitude = AbsMagnitude(simd::LoadI32(input.data() + index));
    lane_sum = AddSatWide(
        lane_sum, ::rund::math32::detail::Widen64(
                      MulUnsignedFixed(magnitude, magnitude)));
  }
  if (index < input.size()) {
    const simd::U32x magnitude = AbsMagnitude(soa::detail::LoadTailI32(input, index));
    lane_sum = AddSatWide(
        lane_sum, ::rund::math32::detail::Widen64(
                      MulUnsignedFixed(magnitude, magnitude)));
  }
  const u64 mean_square = input.empty() ? 0u : ReduceAddSatWide(lane_sum) / static_cast<u64>(input.size());
  return mean_square > static_cast<u64>(static_cast<u32>(FixedMax)) ? FixedMax : static_cast<i32>(mean_square);
}
}  // namespace detail

[[nodiscard]] inline RowStatus RmsNorm(const soa::I32View input,
                                       const soa::I32View weight,
                                       const soa::I32MutView out,
                                       const simd::I32x epsilon) noexcept {
  if (input.size() != weight.size() || input.size() != out.size()) return RowStatus{.processed = 0u, .size_match = false};
  if (input.empty()) return RowStatus{.processed = 0u, .empty_input = true};
  if (simd::Any(simd::Lt(epsilon, simd::SplatI32(0)))) return RowStatus{.processed = 0u, .valid_epsilon = false};
  if (soa::detail::PartiallyOverlaps(input, out) ||
      soa::detail::PartiallyOverlaps(weight, out)) {
    return RowStatus{.processed = 0u, .overlap_ok = false};
  }
  const simd::I32x denominator = Sqrt(AddSat(simd::SplatI32(detail::RowMeanSquare(input)), epsilon));
  u32 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= out.size(); index += simd::LaneCount) {
    const simd::I32x value = simd::LoadI32(input.data() + index);
    const simd::I32x scale = simd::LoadI32(weight.data() + index);
    simd::Store(out.data() + index, DivFixed(MulFixed(value, scale), denominator));
    processed += static_cast<u32>(simd::LaneCount);
  }
  if (index < out.size()) {
    const simd::I32x value = soa::detail::LoadTailI32(input, index);
    const simd::I32x scale = soa::detail::LoadTailI32(weight, index);
    soa::detail::StoreTailI32(out, index, DivFixed(MulFixed(value, scale), denominator));
    processed += ::rund::math32::detail::ScalarSatU32(out.size() - index);
  }
  return RowStatus{.processed = processed};
}
}  // namespace rund::math32::nn
