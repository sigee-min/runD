#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
bezier(const ComputeValue a, const ComputeValue b, const ComputeValue c,
       const ComputeValue t) noexcept {
  return lerp(lerp(a, b, t), lerp(b, c, t), t);
}

[[nodiscard]] inline ComputeValue
bezier(const ComputeValue a, const ComputeValue b, const ComputeValue c,
       const ComputeValue d, const ComputeValue t) noexcept {
  return lerp(bezier(a, b, c, t), bezier(b, c, d, t), t);
}

} // namespace rund::compute_dsl
