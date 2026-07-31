#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue in_range(const ComputeValue value,
                                           const ComputeValue lo,
                                           const ComputeValue hi) noexcept {
  return predicate_and(ge(value, lo), le(value, hi));
}

[[nodiscard]] inline ComputeValue out_range(const ComputeValue value,
                                            const ComputeValue lo,
                                            const ComputeValue hi) noexcept {
  return predicate_not(in_range(value, lo, hi));
}

} // namespace rund::compute_dsl
