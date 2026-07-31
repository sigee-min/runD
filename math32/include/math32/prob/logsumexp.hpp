#pragma once

#include <math32/prob/max.hpp>

namespace rund::math32::prob {
[[nodiscard]] inline LogSumExpResult LogSumExpApprox(const soa::I32View logits) noexcept {
  const MaxResult max = Max(logits);
  if (!max.ok()) return LogSumExpResult{};
  const u64 sum = detail::WeightSumQ1_31(logits, max.value);
  if (sum == 0u) {
    return LogSumExpResult{.value = max.value, .processed = max.processed, .empty_input = false, .zero_sum = true};
  }
  const auto raw = static_cast<::rund::math32::detail::i128>(max.value[0]) +
                   detail::LogWeightsQ5_27ApproxWide(static_cast<::rund::math32::detail::u128>(sum));
  const bool saturated = raw < static_cast<::rund::math32::detail::i128>(FixedMin) ||
                         raw > static_cast<::rund::math32::detail::i128>(FixedMax);
  return LogSumExpResult{.value = simd::SplatI32(detail::ClampWide(raw)),
                         .processed = max.processed,
                         .empty_input = false,
                         .saturated = saturated};
}
}  // namespace rund::math32::prob
