#pragma once

#include <math64/stat/variance.hpp>

namespace rund::math64 {
struct RmsAccumulator {
  simd::U64x squares{};
  u64 processed = 0u;
  bool empty_input = false;
  [[nodiscard]] constexpr bool ok() const noexcept { return !empty_input; }
};

[[nodiscard]] inline RmsAccumulator Rms(const soa::I64View values) noexcept {
  if (values.empty()) return RmsAccumulator{.squares = simd::SplatU64(0), .processed = 0u, .empty_input = true};
  simd::U64x squares = simd::SplatU64(0);
  u64 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= values.size(); index += simd::LaneCount) {
    const simd::U64x magnitude = AbsMagnitude(simd::LoadI64(values.data() + index));
    squares = AddWrapUnsigned(squares, MulUnsignedFixed(magnitude, magnitude));
    processed += static_cast<u64>(simd::LaneCount);
  }
  if (index < values.size()) {
    const simd::U64x magnitude = AbsMagnitude(soa::detail::LoadTailI64(values, index));
    squares = AddWrapUnsigned(squares, MulUnsignedFixed(magnitude, magnitude));
    processed += static_cast<u64>(values.size() - index);
  }
  return RmsAccumulator{.squares = squares, .processed = processed, .empty_input = false};
}
}  // namespace rund::math64
