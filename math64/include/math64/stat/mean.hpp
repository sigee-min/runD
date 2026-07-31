#pragma once

#include <math64/nonlinear/log.hpp>
#include <math64/soa/span.hpp>
#include <math64/soa/tail.hpp>

namespace rund::math64 {
struct MeanAccumulator {
  simd::I64x sum{};
  u64 processed = 0u;
  bool empty_input = false;
  [[nodiscard]] constexpr bool ok() const noexcept { return !empty_input; }
};

[[nodiscard]] inline MeanAccumulator Mean(const soa::I64View values) noexcept {
  if (values.empty()) return MeanAccumulator{.sum = simd::SplatI64(0), .processed = 0u, .empty_input = true};
  simd::I64x sum = simd::SplatI64(0);
  u64 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= values.size(); index += simd::LaneCount) {
    sum = AddWrap(sum, simd::LoadI64(values.data() + index));
    processed += static_cast<u64>(simd::LaneCount);
  }
  if (index < values.size()) {
    sum = AddWrap(sum, soa::detail::LoadTailI64(values, index));
    processed += static_cast<u64>(values.size() - index);
  }
  return MeanAccumulator{.sum = sum, .processed = processed, .empty_input = false};
}
}  // namespace rund::math64
