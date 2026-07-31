#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue CorrelationRatio(
    const ComputeValue numerator, const ComputeValue denom) noexcept {
  return select(eq(denom, 0), fixed_zero(denom),
                div_fixed(numerator, denom));
}

} // namespace rund::compute_dsl::detail
