#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue min(const ComputeValue a,
                                      const ComputeValue b,
                                      const ComputeValue c) noexcept {
  return min(min(a, b), c);
}

[[nodiscard]] inline ComputeValue
min(const ComputeValue a, const ComputeValue b, const ComputeValue c,
    const ComputeValue d) noexcept {
  return min(min(a, b), min(c, d));
}

[[nodiscard]] inline ComputeValue max(const ComputeValue a,
                                      const ComputeValue b,
                                      const ComputeValue c) noexcept {
  return max(max(a, b), c);
}

[[nodiscard]] inline ComputeValue
max(const ComputeValue a, const ComputeValue b, const ComputeValue c,
    const ComputeValue d) noexcept {
  return max(max(a, b), max(c, d));
}

}  // namespace rund::compute_dsl
