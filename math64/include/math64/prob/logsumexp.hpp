#pragma once

#include <math64/prob/max.hpp>

namespace rund::math64::prob {
[[nodiscard]] inline LogSumExpResult LogSumExpApprox(const soa::I64View logits) noexcept {
  const MaxResult max = Max(logits);
  if (!max.ok()) return LogSumExpResult{};
  const ::rund::math64::detail::u128 sum = detail::WeightSumQ1_63(logits, max.value);
  if (sum == 0u) {
    return LogSumExpResult{.value = max.value, .processed = max.processed, .empty_input = false, .zero_sum = true};
  }
  const auto raw = static_cast<::rund::math64::detail::i128>(max.value[0]) +
                   detail::LogWeightsQ5_59ApproxWide(sum);
  const bool saturated = raw < static_cast<::rund::math64::detail::i128>(FixedMin) ||
                         raw > static_cast<::rund::math64::detail::i128>(FixedMax);
  return LogSumExpResult{.value = simd::SplatI64(detail::ClampWide(raw)),
                         .processed = max.processed,
                         .empty_input = false,
                         .saturated = saturated};
}
}  // namespace rund::math64::prob
