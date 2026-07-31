#pragma once

#include <kernel/program/compute/dsl/functions/linear/dot/value.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
conv(const ComputeValue x0, const ComputeValue x1, const ComputeValue x2,
     const ComputeValue k0, const ComputeValue k1,
     const ComputeValue k2) noexcept {
  return dot(x0, x1, x2, k0, k1, k2);
}

[[nodiscard]] inline ComputeValue
conv(const ComputeValue x0, const ComputeValue x1, const ComputeValue x2,
     const ComputeValue x3, const ComputeValue x4, const ComputeValue k0,
     const ComputeValue k1, const ComputeValue k2, const ComputeValue k3,
     const ComputeValue k4) noexcept {
  return dot(x0, x1, x2, x3, x4, k0, k1, k2, k3, k4);
}

[[nodiscard]] inline ComputeValue
conv(const ComputeValue x0, const ComputeValue x1, const ComputeValue x2,
     const ComputeValue x3, const ComputeValue x4, const ComputeValue x5,
     const ComputeValue x6, const ComputeValue k0, const ComputeValue k1,
     const ComputeValue k2, const ComputeValue k3, const ComputeValue k4,
     const ComputeValue k5, const ComputeValue k6) noexcept {
  return add_sat(dot(x0, x1, x2, x3, x4, x5, k0, k1, k2, k3, k4, k5),
                 mul_fixed(x6, k6));
}

[[nodiscard]] inline ComputeValue
conv(const ComputeValue x0, const ComputeValue x1, const ComputeValue x2,
     const ComputeValue x3, const ComputeValue x4, const ComputeValue x5,
     const ComputeValue x6, const ComputeValue x7, const ComputeValue x8,
     const ComputeValue k0, const ComputeValue k1, const ComputeValue k2,
     const ComputeValue k3, const ComputeValue k4, const ComputeValue k5,
     const ComputeValue k6, const ComputeValue k7,
     const ComputeValue k8) noexcept {
  return add_sat(
      dot(x0, x1, x2, x3, x4, x5, x6, x7, k0, k1, k2, k3, k4, k5, k6, k7),
      mul_fixed(x8, k8));
}

} // namespace rund::compute_dsl
