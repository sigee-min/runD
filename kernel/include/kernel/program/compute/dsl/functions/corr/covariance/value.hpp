#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
cov(const ComputeValue x0, const ComputeValue x1, const ComputeValue y0,
    const ComputeValue y1) noexcept {
  const ComputeValue mx = mean(x0, x1);
  const ComputeValue my = mean(y0, y1);
  return mean(mul_fixed(centered(x0, mx), centered(y0, my)),
              mul_fixed(centered(x1, mx), centered(y1, my)));
}

[[nodiscard]] inline ComputeValue
cov(const ComputeValue x0, const ComputeValue x1, const ComputeValue x2,
    const ComputeValue y0, const ComputeValue y1,
    const ComputeValue y2) noexcept {
  const ComputeValue mx = mean(x0, x1, x2);
  const ComputeValue my = mean(y0, y1, y2);
  return mean(mul_fixed(centered(x0, mx), centered(y0, my)),
              mul_fixed(centered(x1, mx), centered(y1, my)),
              mul_fixed(centered(x2, mx), centered(y2, my)));
}

[[nodiscard]] inline ComputeValue
cov(const ComputeValue x0, const ComputeValue x1, const ComputeValue x2,
    const ComputeValue x3, const ComputeValue y0, const ComputeValue y1,
    const ComputeValue y2, const ComputeValue y3) noexcept {
  const ComputeValue mx = mean(x0, x1, x2, x3);
  const ComputeValue my = mean(y0, y1, y2, y3);
  return mean(mul_fixed(centered(x0, mx), centered(y0, my)),
              mul_fixed(centered(x1, mx), centered(y1, my)),
              mul_fixed(centered(x2, mx), centered(y2, my)),
              mul_fixed(centered(x3, mx), centered(y3, my)));
}

} // namespace rund::compute_dsl
