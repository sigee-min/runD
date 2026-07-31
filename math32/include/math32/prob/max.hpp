#pragma once

#include <math32/prob/kernel.hpp>

#include <array>

namespace rund::math32::prob {
[[nodiscard]] inline MaxResult Max(const soa::I32View values) noexcept {
  if (values.empty()) return MaxResult{};
  simd::I32x best = simd::SplatI32(FixedMin);
  simd::U32x best_index = simd::SplatU32(0u);
  u32 processed = 0u;
  std::size_t current = 0u;
  for (; current + simd::LaneCount <= values.size(); current += simd::LaneCount) {
    const simd::I32x candidate = simd::LoadI32(values.data() + current);
    const simd::U32x candidate_index{static_cast<u32>(current), static_cast<u32>(current + 1u),
                                     static_cast<u32>(current + 2u), static_cast<u32>(current + 3u)};
    const simd::Mask32x take = simd::Gt(candidate, best);
    best = simd::Select(take, candidate, best);
    best_index = simd::Select(take, candidate_index, best_index);
    processed += static_cast<u32>(simd::LaneCount);
  }
  if (current < values.size()) {
    const simd::I32x candidate = soa::detail::LoadTailI32(values, current, FixedMin);
    const simd::U32x candidate_index{static_cast<u32>(current), static_cast<u32>(current + 1u),
                                     static_cast<u32>(current + 2u), static_cast<u32>(current + 3u)};
    const simd::Mask32x take = simd::Gt(candidate, best);
    best = simd::Select(take, candidate, best);
    best_index = simd::Select(take, candidate_index, best_index);
    processed += ::rund::math32::detail::ScalarSatU32(values.size() - current);
  }
  alignas(16) std::array<i32, simd::LaneCount> best_lanes{};
  alignas(16) std::array<u32, simd::LaneCount> index_lanes{};
  simd::Store(best_lanes.data(), best);
  simd::Store(index_lanes.data(), best_index);
  i32 row_best = best_lanes[0];
  u32 row_index = index_lanes[0];
  for (std::size_t lane = 1u; lane < simd::LaneCount; ++lane) {
    if (best_lanes[lane] > row_best || (best_lanes[lane] == row_best && index_lanes[lane] < row_index)) {
      row_best = best_lanes[lane];
      row_index = index_lanes[lane];
    }
  }
  return MaxResult{.value = simd::SplatI32(row_best),
                   .index = simd::SplatU32(row_index),
                   .processed = processed,
                   .empty_input = false};
}
}  // namespace rund::math32::prob
