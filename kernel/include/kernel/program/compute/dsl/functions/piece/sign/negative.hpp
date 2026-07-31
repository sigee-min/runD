#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
negative_part(const ComputeValue value) noexcept {
  return min(value, fixed_zero(value));
}

} // namespace rund::compute_dsl
