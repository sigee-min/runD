#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
standard_scale(const ComputeValue lhs, const ComputeValue rhs) noexcept {
  return sqrt(var(lhs, rhs));
}

[[nodiscard]] inline ComputeValue standard_scale(const ComputeValue a,
                                                const ComputeValue b,
                                                const ComputeValue c) noexcept {
  return sqrt(var(a, b, c));
}

[[nodiscard]] inline ComputeValue
standard_scale(const ComputeValue a, const ComputeValue b,
               const ComputeValue c, const ComputeValue d) noexcept {
  return sqrt(var(a, b, c, d));
}

} // namespace rund::compute_dsl::detail
