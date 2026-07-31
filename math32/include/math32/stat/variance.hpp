#pragma once

#include <math32/stat/mean.hpp>

namespace rund::math32 {
struct VarianceAccumulator {
  simd::U32x squares{};
  u32 processed = 0u;
  bool empty_input = false;
  [[nodiscard]] constexpr bool ok() const noexcept { return !empty_input; }
};

[[nodiscard]] inline simd::U32x DifferenceMagnitude(const simd::I32x lhs, const simd::I32x rhs) noexcept {
  return AbsMagnitude(SubWrap(lhs, rhs));
}
[[nodiscard]] inline VarianceAccumulator Variance(const soa::I32View values, const simd::I32x mean) noexcept {
  if (values.empty()) return VarianceAccumulator{.squares = simd::SplatU32(0), .processed = 0u, .empty_input = true};
  simd::U32x squares = simd::SplatU32(0);
  u32 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= values.size(); index += simd::LaneCount) {
    const simd::U32x diff = DifferenceMagnitude(simd::LoadI32(values.data() + index), mean);
    squares = AddWrapUnsigned(squares, MulUnsignedFixed(diff, diff));
    processed += static_cast<u32>(simd::LaneCount);
  }
  if (index < values.size()) {
    const simd::U32x diff = DifferenceMagnitude(soa::detail::LoadTailI32Or(values, index, mean), mean);
    squares = AddWrapUnsigned(squares, MulUnsignedFixed(diff, diff));
    processed += ::rund::math32::detail::ScalarSatU32(values.size() - index);
  }
  return VarianceAccumulator{.squares = squares, .processed = processed, .empty_input = false};
}
}  // namespace rund::math32
