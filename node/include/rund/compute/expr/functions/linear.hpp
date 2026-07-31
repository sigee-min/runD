#pragma once

#include <rund/compute/expr/functions/core.hpp>

namespace rund::compute {

[[nodiscard]] constexpr auto dot(const auto &ax, const auto &ay, const auto &bx,
                                 const auto &by) {
  return add_sat(mul_fixed(ax, bx), mul_fixed(ay, by));
}

[[nodiscard]] constexpr auto dot(const auto &ax, const auto &ay, const auto &az,
                                 const auto &bx, const auto &by,
                                 const auto &bz) {
  return add_sat(dot(ax, ay, bx, by), mul_fixed(az, bz));
}

[[nodiscard]] constexpr auto dot(const auto &a0, const auto &a1, const auto &a2,
                                 const auto &a3, const auto &b0, const auto &b1,
                                 const auto &b2, const auto &b3) {
  return add_sat(dot(a0, a1, b0, b1), dot(a2, a3, b2, b3));
}

[[nodiscard]] constexpr auto dot(const auto &a0, const auto &a1, const auto &a2,
                                 const auto &a3, const auto &a4, const auto &b0,
                                 const auto &b1, const auto &b2, const auto &b3,
                                 const auto &b4) {
  return add_sat(dot(a0, a1, a2, a3, b0, b1, b2, b3), mul_fixed(a4, b4));
}

[[nodiscard]] constexpr auto dot(const auto &a0, const auto &a1, const auto &a2,
                                 const auto &a3, const auto &a4, const auto &a5,
                                 const auto &b0, const auto &b1, const auto &b2,
                                 const auto &b3, const auto &b4,
                                 const auto &b5) {
  return add_sat(dot(a0, a1, a2, b0, b1, b2), dot(a3, a4, a5, b3, b4, b5));
}

[[nodiscard]] constexpr auto dot(const auto &a0, const auto &a1, const auto &a2,
                                 const auto &a3, const auto &a4, const auto &a5,
                                 const auto &a6, const auto &a7, const auto &b0,
                                 const auto &b1, const auto &b2, const auto &b3,
                                 const auto &b4, const auto &b5, const auto &b6,
                                 const auto &b7) {
  return add_sat(dot(a0, a1, a2, a3, b0, b1, b2, b3),
                 dot(a4, a5, a6, a7, b4, b5, b6, b7));
}

[[nodiscard]] constexpr auto conv(const auto &x0, const auto &x1,
                                  const auto &x2, const auto &k0,
                                  const auto &k1, const auto &k2) {
  return dot(x0, x1, x2, k0, k1, k2);
}
[[nodiscard]] constexpr auto conv(const auto &x0, const auto &x1,
                                  const auto &x2, const auto &x3,
                                  const auto &x4, const auto &k0,
                                  const auto &k1, const auto &k2,
                                  const auto &k3, const auto &k4) {
  return dot(x0, x1, x2, x3, x4, k0, k1, k2, k3, k4);
}
[[nodiscard]] constexpr auto
conv(const auto &x0, const auto &x1, const auto &x2, const auto &x3,
     const auto &x4, const auto &x5, const auto &x6, const auto &k0,
     const auto &k1, const auto &k2, const auto &k3, const auto &k4,
     const auto &k5, const auto &k6) {
  return add_sat(dot(x0, x1, x2, x3, x4, x5, k0, k1, k2, k3, k4, k5),
                 mul_fixed(x6, k6));
}
[[nodiscard]] constexpr auto
conv(const auto &x0, const auto &x1, const auto &x2, const auto &x3,
     const auto &x4, const auto &x5, const auto &x6, const auto &x7,
     const auto &x8, const auto &k0, const auto &k1, const auto &k2,
     const auto &k3, const auto &k4, const auto &k5, const auto &k6,
     const auto &k7, const auto &k8) {
  return add_sat(
      dot(x0, x1, x2, x3, x4, x5, x6, x7, k0, k1, k2, k3, k4, k5, k6, k7),
      mul_fixed(x8, k8));
}

} // namespace rund::compute
