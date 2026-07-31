#pragma once

#include <math64/stat/mean.hpp>

namespace rund::math64 {
struct VarianceAccumulator {
  simd::U64x squares{};
  u64 processed = 0u;
  bool empty_input = false;
  [[nodiscard]] constexpr bool ok() const noexcept { return !empty_input; }
};

[[nodiscard]] inline simd::U64x DifferenceMagnitude(const simd::I64x lhs, const simd::I64x rhs) noexcept {
  return AbsMagnitude(SubWrap(lhs, rhs));
}
[[nodiscard]] inline VarianceAccumulator Variance(const soa::I64View values, const simd::I64x mean) noexcept {
  if (values.empty()) return VarianceAccumulator{.squares = simd::SplatU64(0), .processed = 0u, .empty_input = true};
  simd::U64x squares = simd::SplatU64(0);
  u64 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= values.size(); index += simd::LaneCount) {
    const simd::U64x diff = DifferenceMagnitude(simd::LoadI64(values.data() + index), mean);
    squares = AddWrapUnsigned(squares, MulUnsignedFixed(diff, diff));
    processed += static_cast<u64>(simd::LaneCount);
  }
  if (index < values.size()) {
    const simd::U64x diff = DifferenceMagnitude(soa::detail::LoadTailI64Or(values, index, mean), mean);
    squares = AddWrapUnsigned(squares, MulUnsignedFixed(diff, diff));
    processed += static_cast<u64>(values.size() - index);
  }
  return VarianceAccumulator{.squares = squares, .processed = processed, .empty_input = false};
}
}  // namespace rund::math64
