#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue deadzone(const ComputeValue value,
                                           const ComputeValue tol) noexcept {
  return select(near(value, tol), fixed_zero(value), value);
}

[[nodiscard]] inline ComputeValue snap(const ComputeValue value,
                                       const ComputeValue target,
                                       const ComputeValue tol) noexcept {
  return select(near(value, target, tol), target, value);
}

} // namespace rund::compute_dsl
