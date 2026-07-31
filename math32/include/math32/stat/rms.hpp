#pragma once

#include <math32/stat/variance.hpp>

namespace rund::math32 {
struct RmsAccumulator {
  simd::U32x squares{};
  u32 processed = 0u;
  bool empty_input = false;
  [[nodiscard]] constexpr bool ok() const noexcept { return !empty_input; }
};

[[nodiscard]] inline RmsAccumulator Rms(const soa::I32View values) noexcept {
  if (values.empty()) return RmsAccumulator{.squares = simd::SplatU32(0), .processed = 0u, .empty_input = true};
  simd::U32x squares = simd::SplatU32(0);
  u32 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= values.size(); index += simd::LaneCount) {
    const simd::U32x magnitude = AbsMagnitude(simd::LoadI32(values.data() + index));
    squares = AddWrapUnsigned(squares, MulUnsignedFixed(magnitude, magnitude));
    processed += static_cast<u32>(simd::LaneCount);
  }
  if (index < values.size()) {
    const simd::U32x magnitude = AbsMagnitude(soa::detail::LoadTailI32(values, index));
    squares = AddWrapUnsigned(squares, MulUnsignedFixed(magnitude, magnitude));
    processed += ::rund::math32::detail::ScalarSatU32(values.size() - index);
  }
  return RmsAccumulator{.squares = squares, .processed = processed, .empty_input = false};
}
}  // namespace rund::math32
