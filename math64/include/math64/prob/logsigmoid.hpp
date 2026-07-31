#pragma once

#include <math64/prob/softplus.hpp>

namespace rund::math64::prob {
[[nodiscard]] inline simd::I64x LogSigmoidApprox(const simd::I64x value) noexcept {
  return SubSat(value, SoftplusApprox(value));
}
}  // namespace rund::math64::prob
