#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
clamp_range(const ComputeValue value, const ComputeValue a,
            const ComputeValue b) noexcept {
  return clamp(value, min(a, b), max(a, b));
}

} // namespace rund::compute_dsl
