#pragma once

#include <math64/prob/logsumexp.hpp>

namespace rund::math64::prob {
[[nodiscard]] inline CrossEntropyResult CrossEntropyLogitsApprox(const soa::I64View logits, const std::size_t target_index) noexcept {
  if (logits.empty()) return CrossEntropyResult{};
  if (target_index >= logits.size()) return CrossEntropyResult{.empty_input = false, .valid_target = false};
  const LogSumExpResult log_sum = LogSumExpApprox(logits);
  const simd::I64x target = simd::SplatI64(logits[target_index]);
  return CrossEntropyResult{.value = SubSat(log_sum.value, target),
                            .processed = log_sum.processed,
                            .empty_input = false,
                            .valid_target = true,
                            .zero_sum = log_sum.zero_sum,
                            .saturated = log_sum.saturated};
}
}  // namespace rund::math64::prob
