#pragma once

#include <math64/prob/softmax.hpp>

namespace rund::math64::prob {
[[nodiscard]] inline RowStatus LogSoftmaxApprox(const soa::I64View logits, const soa::I64MutView out) noexcept {
  RowStatus status = detail::ValidateRow(logits, out);
  if (!status.ok() || status.empty_input) return status;
  const LogSumExpResult log_sum = LogSumExpApprox(logits);
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= logits.size(); index += simd::LaneCount) {
    simd::Store(out.data() + index, SubSat(simd::LoadI64(logits.data() + index), log_sum.value));
  }
  if (index < logits.size()) {
    soa::detail::StoreTailI64(out, index, SubSat(soa::detail::LoadTailI64(logits, index), log_sum.value));
  }
  return status;
}
}  // namespace rund::math64::prob
