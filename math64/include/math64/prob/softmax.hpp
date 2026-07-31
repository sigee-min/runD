#pragma once

#include <math64/prob/logsumexp.hpp>

namespace rund::math64::prob {
[[nodiscard]] inline RowStatus SoftmaxApprox(const soa::I64View logits, const soa::I64MutView out) noexcept {
  RowStatus status = detail::ValidateRow(logits, out);
  if (!status.ok() || status.empty_input) return status;
  const MaxResult max = Max(logits);
  const ::rund::math64::detail::u128 denominator = detail::WeightSumQ1_63(logits, max.value);
  if (denominator == 0u) return RowStatus{.processed = status.processed, .zero_sum = true};
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= logits.size(); index += simd::LaneCount) {
    const simd::I64x weights = detail::ExpWeights(logits, index, max.value, false);
    simd::Store(out.data() + index, detail::NormalizeWeights(weights, denominator));
  }
  if (index < logits.size()) {
    const simd::I64x weights = detail::ExpWeights(logits, index, max.value, true);
    soa::detail::StoreTailI64(out, index, detail::NormalizeWeights(weights, denominator));
  }
  return status;
}
}  // namespace rund::math64::prob
