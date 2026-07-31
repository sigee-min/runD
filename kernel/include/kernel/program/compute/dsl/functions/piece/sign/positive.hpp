#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
positive_part(const ComputeValue value) noexcept {
  return max(value, fixed_zero(value));
}

} // namespace rund::compute_dsl
