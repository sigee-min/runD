#pragma once

#include <math32/prob/softmax.hpp>

namespace rund::math32::prob {
[[nodiscard]] inline RowStatus LogSoftmaxApprox(const soa::I32View logits, const soa::I32MutView out) noexcept {
  RowStatus status = detail::ValidateRow(logits, out);
  if (!status.ok() || status.empty_input) return status;
  const LogSumExpResult log_sum = LogSumExpApprox(logits);
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= logits.size(); index += simd::LaneCount) {
    simd::Store(out.data() + index, SubSat(simd::LoadI32(logits.data() + index), log_sum.value));
  }
  if (index < logits.size()) {
    soa::detail::StoreTailI32(out, index, SubSat(soa::detail::LoadTailI32(logits, index), log_sum.value));
  }
  return status;
}
}  // namespace rund::math32::prob
