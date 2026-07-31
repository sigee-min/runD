#pragma once

#include <rund/compute/expr/functions/geometry.hpp>

namespace rund::compute {

struct MatOp final {
  struct DeterminantTag final {};
  struct SolveTag final {};
  struct TraceTag final {};
  struct TransposeTag final {};

  inline static constexpr DeterminantTag Determinant{};
  inline static constexpr SolveTag Solve{};
  inline static constexpr TraceTag Trace{};
  inline static constexpr TransposeTag Transpose{};
};

[[nodiscard]] constexpr auto mat(MatOp::DeterminantTag, const auto &m00,
                                 const auto &m01, const auto &m10,
                                 const auto &m11) {
  return sub_sat(mul_fixed(m00, m11), mul_fixed(m01, m10));
}
[[nodiscard]] constexpr auto mat(MatOp::DeterminantTag, const auto &m00,
                                 const auto &m01, const auto &m02,
                                 const auto &m10, const auto &m11,
                                 const auto &m12, const auto &m20,
                                 const auto &m21, const auto &m22) {
  const auto a = mul_fixed(m00, mat(MatOp::Determinant, m11, m12, m21, m22));
  const auto b = mul_fixed(m01, mat(MatOp::Determinant, m10, m12, m20, m22));
  const auto c = mul_fixed(m02, mat(MatOp::Determinant, m10, m11, m20, m21));
  return add_sat(sub_sat(a, b), c);
}
[[nodiscard]] constexpr auto mat(Axis::XTag, const auto &m00, const auto &m01,
                                 const auto &x, const auto &y) {
  return dot(m00, m01, x, y);
}
[[nodiscard]] constexpr auto mat(Axis::YTag, const auto &m10, const auto &m11,
                                 const auto &x, const auto &y) {
  return dot(m10, m11, x, y);
}
[[nodiscard]] constexpr auto mat(Axis::XTag, const auto &m00, const auto &m01,
                                 const auto &m02, const auto &x, const auto &y,
                                 const auto &z) {
  return dot(m00, m01, m02, x, y, z);
}
[[nodiscard]] constexpr auto mat(Axis::YTag, const auto &m10, const auto &m11,
                                 const auto &m12, const auto &x, const auto &y,
                                 const auto &z) {
  return dot(m10, m11, m12, x, y, z);
}
[[nodiscard]] constexpr auto mat(Axis::ZTag, const auto &m20, const auto &m21,
                                 const auto &m22, const auto &x, const auto &y,
                                 const auto &z) {
  return dot(m20, m21, m22, x, y, z);
}
[[nodiscard]] constexpr auto mat(MatOp::SolveTag, Axis::XTag, const auto &m00,
                                 const auto &m01, const auto &m10,
                                 const auto &m11, const auto &b0,
                                 const auto &b1) {
  return ratio(mat(MatOp::Determinant, b0, m01, b1, m11),
               mat(MatOp::Determinant, m00, m01, m10, m11));
}
[[nodiscard]] constexpr auto mat(MatOp::SolveTag, Axis::YTag, const auto &m00,
                                 const auto &m01, const auto &m10,
                                 const auto &m11, const auto &b0,
                                 const auto &b1) {
  return ratio(mat(MatOp::Determinant, m00, b0, m10, b1),
               mat(MatOp::Determinant, m00, m01, m10, m11));
}
[[nodiscard]] constexpr auto mat(MatOp::TraceTag, const auto &m00,
                                 const auto &m11) {
  return add_sat(m00, m11);
}
[[nodiscard]] constexpr auto mat(MatOp::TraceTag, const auto &m00,
                                 const auto &m11, const auto &m22) {
  return add_sat(add_sat(m00, m11), m22);
}
[[nodiscard]] constexpr auto mat(MatOp::TransposeTag, Axis::XTag,
                                 const auto &m00, const auto &m10,
                                 const auto &x, const auto &y) {
  return dot(m00, m10, x, y);
}
[[nodiscard]] constexpr auto mat(MatOp::TransposeTag, Axis::YTag,
                                 const auto &m01, const auto &m11,
                                 const auto &x, const auto &y) {
  return dot(m01, m11, x, y);
}
[[nodiscard]] constexpr auto mat(MatOp::TransposeTag, Axis::XTag,
                                 const auto &m00, const auto &m10,
                                 const auto &m20, const auto &x, const auto &y,
                                 const auto &z) {
  return dot(m00, m10, m20, x, y, z);
}
[[nodiscard]] constexpr auto mat(MatOp::TransposeTag, Axis::YTag,
                                 const auto &m01, const auto &m11,
                                 const auto &m21, const auto &x, const auto &y,
                                 const auto &z) {
  return dot(m01, m11, m21, x, y, z);
}
[[nodiscard]] constexpr auto mat(MatOp::TransposeTag, Axis::ZTag,
                                 const auto &m02, const auto &m12,
                                 const auto &m22, const auto &x, const auto &y,
                                 const auto &z) {
  return dot(m02, m12, m22, x, y, z);
}

[[nodiscard]] constexpr auto aff(Axis::XTag axis, const auto &m00,
                                 const auto &m01, const auto &offset,
                                 const auto &x, const auto &y) {
  return add_sat(mat(axis, m00, m01, x, y), offset);
}
[[nodiscard]] constexpr auto aff(Axis::YTag axis, const auto &m10,
                                 const auto &m11, const auto &offset,
                                 const auto &x, const auto &y) {
  return add_sat(mat(axis, m10, m11, x, y), offset);
}
[[nodiscard]] constexpr auto aff(Axis::XTag axis, const auto &m00,
                                 const auto &m01, const auto &m02,
                                 const auto &offset, const auto &x,
                                 const auto &y, const auto &z) {
  return add_sat(mat(axis, m00, m01, m02, x, y, z), offset);
}
[[nodiscard]] constexpr auto aff(Axis::YTag axis, const auto &m10,
                                 const auto &m11, const auto &m12,
                                 const auto &offset, const auto &x,
                                 const auto &y, const auto &z) {
  return add_sat(mat(axis, m10, m11, m12, x, y, z), offset);
}
[[nodiscard]] constexpr auto aff(Axis::ZTag axis, const auto &m20,
                                 const auto &m21, const auto &m22,
                                 const auto &offset, const auto &x,
                                 const auto &y, const auto &z) {
  return add_sat(mat(axis, m20, m21, m22, x, y, z), offset);
}

[[nodiscard]] constexpr auto mix(const auto &a, const auto &b, const auto &wa,
                                 const auto &wb) {
  return add_sat(mul_fixed(a, wa), mul_fixed(b, wb));
}
[[nodiscard]] constexpr auto mix(const auto &a, const auto &b, const auto &c,
                                 const auto &wa, const auto &wb,
                                 const auto &wc) {
  return add_sat(mix(a, b, wa, wb), mul_fixed(c, wc));
}
[[nodiscard]] constexpr auto mix(const auto &a, const auto &b, const auto &c,
                                 const auto &d, const auto &wa, const auto &wb,
                                 const auto &wc, const auto &wd) {
  return add_sat(mix(a, b, wa, wb), mix(c, d, wc, wd));
}
[[nodiscard]] constexpr auto weighted_mean(const auto &a, const auto &b,
                                           const auto &wa, const auto &wb) {
  return ratio(mix(a, b, wa, wb), sum(wa, wb));
}
[[nodiscard]] constexpr auto weighted_mean(const auto &a, const auto &b,
                                           const auto &c, const auto &wa,
                                           const auto &wb, const auto &wc) {
  return ratio(mix(a, b, c, wa, wb, wc), sum(wa, wb, wc));
}
[[nodiscard]] constexpr auto weighted_mean(const auto &a, const auto &b,
                                           const auto &c, const auto &d,
                                           const auto &wa, const auto &wb,
                                           const auto &wc, const auto &wd) {
  return ratio(mix(a, b, c, d, wa, wb, wc, wd), sum(wa, wb, wc, wd));
}

[[nodiscard]] constexpr auto poly(const auto &x, const auto &c0, const auto &c1,
                                  const auto &c2) {
  return add_sat(c0, mul_fixed(x, add_sat(c1, mul_fixed(c2, x))));
}
[[nodiscard]] constexpr auto poly(const auto &x, const auto &c0, const auto &c1,
                                  const auto &c2, const auto &c3) {
  return add_sat(
      c0,
      mul_fixed(x, add_sat(c1, mul_fixed(x, add_sat(c2, mul_fixed(c3, x))))));
}
[[nodiscard]] constexpr auto poly_deriv(const auto &x, const auto &c1,
                                        const auto &c2) {
  return add_sat(c1, mul_fixed(x, add_sat(c2, c2)));
}
[[nodiscard]] constexpr auto poly_deriv(const auto &x, const auto &c1,
                                        const auto &c2, const auto &c3) {
  const auto two_c2 = add_sat(c2, c2);
  const auto three_c3 = add_sat(add_sat(c3, c3), c3);
  return add_sat(c1, mul_fixed(x, add_sat(two_c2, mul_fixed(three_c3, x))));
}

} // namespace rund::compute
