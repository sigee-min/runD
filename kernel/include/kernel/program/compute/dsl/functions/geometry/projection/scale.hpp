#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
ProjectScale(const ComputeValue dot_value,
             const ComputeValue denom) noexcept {
  return select(eq(denom, 0), fixed_zero(dot_value),
                div_fixed(dot_value, denom));
}

} // namespace rund::compute_dsl::detail
