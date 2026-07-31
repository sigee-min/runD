#pragma once

#include <math64/prob/kernel.hpp>

#include <array>

namespace rund::math64::prob {
[[nodiscard]] inline MaxResult Max(const soa::I64View values) noexcept {
  if (values.empty()) return MaxResult{};
  simd::I64x best = simd::SplatI64(FixedMin);
  simd::U64x best_index = simd::SplatU64(0u);
  u64 processed = 0u;
  std::size_t current = 0u;
  for (; current + simd::LaneCount <= values.size(); current += simd::LaneCount) {
    const simd::I64x candidate = simd::LoadI64(values.data() + current);
    const simd::U64x candidate_index{static_cast<u64>(current), static_cast<u64>(current + 1u)};
    const simd::Mask64x take = simd::Gt(candidate, best);
    best = simd::Select(take, candidate, best);
    best_index = simd::Select(take, candidate_index, best_index);
    processed += static_cast<u64>(simd::LaneCount);
  }
  if (current < values.size()) {
    const simd::I64x candidate = soa::detail::LoadTailI64(values, current, FixedMin);
    const simd::U64x candidate_index{static_cast<u64>(current), static_cast<u64>(current + 1u)};
    const simd::Mask64x take = simd::Gt(candidate, best);
    best = simd::Select(take, candidate, best);
    best_index = simd::Select(take, candidate_index, best_index);
    processed += ::rund::math64::detail::ScalarSatU32(values.size() - current);
  }
  alignas(16) std::array<i64, simd::LaneCount> best_lanes{};
  alignas(16) std::array<u64, simd::LaneCount> index_lanes{};
  simd::Store(best_lanes.data(), best);
  simd::Store(index_lanes.data(), best_index);
  i64 row_best = best_lanes[0];
  u64 row_index = index_lanes[0];
  for (std::size_t lane = 1u; lane < simd::LaneCount; ++lane) {
    if (best_lanes[lane] > row_best || (best_lanes[lane] == row_best && index_lanes[lane] < row_index)) {
      row_best = best_lanes[lane];
      row_index = index_lanes[lane];
    }
  }
  return MaxResult{.value = simd::SplatI64(row_best),
                   .index = simd::SplatU64(row_index),
                   .processed = processed,
                   .empty_input = false};
}
}  // namespace rund::math64::prob
