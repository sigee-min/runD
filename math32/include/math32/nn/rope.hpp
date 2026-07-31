#pragma once

#include <math32/nn/norm.hpp>
#include <math32/soa/range.hpp>

namespace rund::math32::nn {
[[nodiscard]] inline RopePairResult RopePair(const simd::I32x even,
                                             const simd::I32x odd,
                                             const simd::I32x sin,
                                             const simd::I32x cos) noexcept {
  return RopePairResult{.even = SubSat(MulFixed(even, cos), MulFixed(odd, sin)),
                        .odd = AddSat(MulFixed(even, sin), MulFixed(odd, cos))};
}
[[nodiscard]] inline RowStatus RopeRow(const soa::I32View even,
                                       const soa::I32View odd,
                                       const soa::I32View sin,
                                       const soa::I32View cos,
                                       const soa::I32MutView out_even,
                                       const soa::I32MutView out_odd) noexcept {
  const bool size_match = even.size() == odd.size() && even.size() == sin.size() && even.size() == cos.size() &&
                          even.size() == out_even.size() && even.size() == out_odd.size();
  if (!size_match) return RowStatus{.processed = 0u, .size_match = false};
  if (even.empty()) return RowStatus{.processed = 0u, .empty_input = true};
  if (soa::detail::PartiallyOverlaps(even, out_even) ||
      soa::detail::PartiallyOverlaps(odd, out_odd) ||
      soa::detail::Overlaps(sin, out_even) ||
      soa::detail::Overlaps(cos, out_odd)) {
    return RowStatus{.processed = 0u, .overlap_ok = false};
  }
  u32 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= even.size(); index += simd::LaneCount) {
    const RopePairResult rotated = RopePair(simd::LoadI32(even.data() + index),
                                            simd::LoadI32(odd.data() + index),
                                            simd::LoadI32(sin.data() + index),
                                            simd::LoadI32(cos.data() + index));
    simd::Store(out_even.data() + index, rotated.even);
    simd::Store(out_odd.data() + index, rotated.odd);
    processed += static_cast<u32>(simd::LaneCount);
  }
  if (index < even.size()) {
    const RopePairResult rotated = RopePair(soa::detail::LoadTailI32(even, index),
                                            soa::detail::LoadTailI32(odd, index),
                                            soa::detail::LoadTailI32(sin, index),
                                            soa::detail::LoadTailI32(cos, index));
    soa::detail::StoreTailI32(out_even, index, rotated.even);
    soa::detail::StoreTailI32(out_odd, index, rotated.odd);
    processed += ::rund::math32::detail::ScalarSatU32(even.size() - index);
  }
  return RowStatus{.processed = processed};
}
}  // namespace rund::math32::nn
