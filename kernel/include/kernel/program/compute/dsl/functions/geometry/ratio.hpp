#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
GeometryRatio(const ComputeValue numerator,
              const ComputeValue denominator) noexcept {
  return select(eq(denominator, 0), fixed_zero(numerator),
                div_fixed(numerator, denominator));
}

} // namespace rund::compute_dsl::detail
