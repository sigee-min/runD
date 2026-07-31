#pragma once

#include <math32/prob/logsumexp.hpp>

namespace rund::math32::prob {
[[nodiscard]] inline RowStatus SoftmaxApprox(const soa::I32View logits, const soa::I32MutView out) noexcept {
  RowStatus status = detail::ValidateRow(logits, out);
  if (!status.ok() || status.empty_input) return status;
  const MaxResult max = Max(logits);
  const u64 denominator = detail::WeightSumQ1_31(logits, max.value);
  if (denominator == 0u) return RowStatus{.processed = status.processed, .zero_sum = true};
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= logits.size(); index += simd::LaneCount) {
    const simd::I32x weights = detail::ExpWeights(logits, index, max.value, false);
    simd::Store(out.data() + index, detail::NormalizeWeights(weights, denominator));
  }
  if (index < logits.size()) {
    const simd::I32x weights = detail::ExpWeights(logits, index, max.value, true);
    soa::detail::StoreTailI32(out, index, detail::NormalizeWeights(weights, denominator));
  }
  return status;
}
}  // namespace rund::math32::prob
