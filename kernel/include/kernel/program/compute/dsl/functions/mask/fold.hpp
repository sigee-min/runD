#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue all(const ComputeValue a,
                                      const ComputeValue b,
                                      const ComputeValue c) noexcept {
  return predicate_and(predicate_and(a, b), c);
}

[[nodiscard]] inline ComputeValue any(const ComputeValue a,
                                      const ComputeValue b,
                                      const ComputeValue c) noexcept {
  return predicate_or(predicate_or(a, b), c);
}

} // namespace rund::compute_dsl
