#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue sum(const ComputeValue lhs,
                                      const ComputeValue rhs) noexcept {
  return add_sat(lhs, rhs);
}

[[nodiscard]] inline ComputeValue sum(const ComputeValue a,
                                      const ComputeValue b,
                                      const ComputeValue c) noexcept {
  return add_sat(sum(a, b), c);
}

[[nodiscard]] inline ComputeValue
sum(const ComputeValue a, const ComputeValue b, const ComputeValue c,
    const ComputeValue d) noexcept {
  return add_sat(sum(a, b), sum(c, d));
}

} // namespace rund::compute_dsl
