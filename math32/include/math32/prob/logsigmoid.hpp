#pragma once

#include <math32/prob/softplus.hpp>

namespace rund::math32::prob {
[[nodiscard]] inline simd::I32x LogSigmoidApprox(const simd::I32x value) noexcept {
  return SubSat(value, SoftplusApprox(value));
}
}  // namespace rund::math32::prob
