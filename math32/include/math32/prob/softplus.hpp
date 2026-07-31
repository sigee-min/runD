#pragma once

#include <math32/prob/kernel.hpp>

namespace rund::math32::prob {
[[nodiscard]] inline simd::I32x SoftplusApprox(const simd::I32x value) noexcept {
  const simd::I32x positive = ::rund::math32::Max(value, simd::SplatI32(0));
  const simd::I32x decay = detail::ExpNegQ5_27ToQ1_31Approx(-Abs(value));
  return AddSat(positive, decay >> 1u);
}
}  // namespace rund::math32::prob
