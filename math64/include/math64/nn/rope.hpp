#pragma once

#include <math64/nn/norm.hpp>
#include <math64/soa/range.hpp>

namespace rund::math64::nn {
[[nodiscard]] inline RopePairResult RopePair(const simd::I64x even,
                                             const simd::I64x odd,
                                             const simd::I64x sin,
                                             const simd::I64x cos) noexcept {
  return RopePairResult{.even = SubSat(MulFixed(even, cos), MulFixed(odd, sin)),
                        .odd = AddSat(MulFixed(even, sin), MulFixed(odd, cos))};
}
[[nodiscard]] inline RowStatus RopeRow(const soa::I64View even,
                                       const soa::I64View odd,
                                       const soa::I64View sin,
                                       const soa::I64View cos,
                                       const soa::I64MutView out_even,
                                       const soa::I64MutView out_odd) noexcept {
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
  u64 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= even.size(); index += simd::LaneCount) {
    const RopePairResult rotated = RopePair(simd::LoadI64(even.data() + index),
                                            simd::LoadI64(odd.data() + index),
                                            simd::LoadI64(sin.data() + index),
                                            simd::LoadI64(cos.data() + index));
    simd::Store(out_even.data() + index, rotated.even);
    simd::Store(out_odd.data() + index, rotated.odd);
    processed += static_cast<u64>(simd::LaneCount);
  }
  if (index < even.size()) {
    const RopePairResult rotated = RopePair(soa::detail::LoadTailI64(even, index),
                                            soa::detail::LoadTailI64(odd, index),
                                            soa::detail::LoadTailI64(sin, index),
                                            soa::detail::LoadTailI64(cos, index));
    soa::detail::StoreTailI64(out_even, index, rotated.even);
    soa::detail::StoreTailI64(out_odd, index, rotated.odd);
    processed += ::rund::math64::detail::ScalarSatU32(even.size() - index);
  }
  return RowStatus{.processed = processed};
}
}  // namespace rund::math64::nn
