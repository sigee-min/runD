#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue near(const ComputeValue value,
                                       const ComputeValue target,
                                       const ComputeValue tol) noexcept {
  return le(absdiff(value, target), abs(tol));
}

[[nodiscard]] inline ComputeValue near(const ComputeValue value,
                                       const ComputeValue tol) noexcept {
  return near(value, fixed_zero(value), tol);
}

} // namespace rund::compute_dsl
