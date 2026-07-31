#pragma once

#include <math32/nonlinear/log.hpp>
#include <math32/soa/span.hpp>
#include <math32/soa/tail.hpp>

namespace rund::math32 {
struct MeanAccumulator {
  simd::I32x sum{};
  u32 processed = 0u;
  bool empty_input = false;
  [[nodiscard]] constexpr bool ok() const noexcept { return !empty_input; }
};

[[nodiscard]] inline MeanAccumulator Mean(const soa::I32View values) noexcept {
  if (values.empty()) return MeanAccumulator{.sum = simd::SplatI32(0), .processed = 0u, .empty_input = true};
  simd::I32x sum = simd::SplatI32(0);
  u32 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= values.size(); index += simd::LaneCount) {
    sum = AddWrap(sum, simd::LoadI32(values.data() + index));
    processed += static_cast<u32>(simd::LaneCount);
  }
  if (index < values.size()) {
    sum = AddWrap(sum, soa::detail::LoadTailI32(values, index));
    processed += ::rund::math32::detail::ScalarSatU32(values.size() - index);
  }
  return MeanAccumulator{.sum = sum, .processed = processed, .empty_input = false};
}
}  // namespace rund::math32
