#pragma once

#include <math64/prob/kernel.hpp>

namespace rund::math64::prob {
[[nodiscard]] inline simd::I64x SoftplusApprox(const simd::I64x value) noexcept {
  const simd::I64x positive = ::rund::math64::Max(value, simd::SplatI64(0));
  const simd::I64x decay = detail::ExpNegQ5_59ToQ1_63Approx(-Abs(value));
  return AddSat(positive, decay >> 1u);
}
}  // namespace rund::math64::prob
