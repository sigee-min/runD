#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue clip(const ComputeValue value,
                                       const ComputeValue limit) noexcept {
  const ComputeValue bound = detail::StorageQuantize(abs(limit));
  return clamp(value, neg_positive_fixed(bound), bound);
}

} // namespace rund::compute_dsl
