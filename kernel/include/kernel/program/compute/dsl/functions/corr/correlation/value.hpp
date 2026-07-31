#pragma once

#include <kernel/program/compute/dsl/functions/corr/correlation/ratio.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
corr(const ComputeValue x0, const ComputeValue x1, const ComputeValue y0,
     const ComputeValue y1) noexcept {
  const ComputeValue denom = sqrt(mul_fixed(var(x0, x1), var(y0, y1)));
  return detail::CorrelationRatio(cov(x0, x1, y0, y1), denom);
}

[[nodiscard]] inline ComputeValue
corr(const ComputeValue x0, const ComputeValue x1, const ComputeValue x2,
     const ComputeValue y0, const ComputeValue y1,
     const ComputeValue y2) noexcept {
  const ComputeValue denom =
      sqrt(mul_fixed(var(x0, x1, x2), var(y0, y1, y2)));
  return detail::CorrelationRatio(cov(x0, x1, x2, y0, y1, y2), denom);
}

[[nodiscard]] inline ComputeValue
corr(const ComputeValue x0, const ComputeValue x1, const ComputeValue x2,
     const ComputeValue x3, const ComputeValue y0, const ComputeValue y1,
     const ComputeValue y2, const ComputeValue y3) noexcept {
  const ComputeValue denom =
      sqrt(mul_fixed(var(x0, x1, x2, x3), var(y0, y1, y2, y3)));
  return detail::CorrelationRatio(cov(x0, x1, x2, x3, y0, y1, y2, y3), denom);
}

} // namespace rund::compute_dsl
