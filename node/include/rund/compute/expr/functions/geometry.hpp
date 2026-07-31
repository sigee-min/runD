#pragma once

#include <rund/compute/expr/functions/metric.hpp>

namespace rund::compute {

struct Axis final {
  struct XTag final {};
  struct YTag final {};
  struct ZTag final {};
  inline static constexpr XTag X{};
  inline static constexpr YTag Y{};
  inline static constexpr ZTag Z{};
};
struct GeometryOp final {
  struct DistanceTag final {};
  struct ParameterTag final {};
  struct ProjectionTag final {};

  inline static constexpr DistanceTag Distance{};
  inline static constexpr ParameterTag Parameter{};
  inline static constexpr ProjectionTag Projection{};
};

[[nodiscard]] constexpr auto proportion(const auto &value, const auto &total) {
  return saturate(ratio(value, total));
}
[[nodiscard]] constexpr auto proportion(Axis::XTag, const auto &x,
                                        const auto &y) {
  return proportion(x, sum(x, y));
}
[[nodiscard]] constexpr auto proportion(Axis::YTag, const auto &x,
                                        const auto &y) {
  return proportion(y, sum(x, y));
}
[[nodiscard]] constexpr auto proportion(Axis::XTag, const auto &x,
                                        const auto &y, const auto &z) {
  return proportion(x, sum(x, y, z));
}
[[nodiscard]] constexpr auto proportion(Axis::YTag, const auto &x,
                                        const auto &y, const auto &z) {
  return proportion(y, sum(x, y, z));
}
[[nodiscard]] constexpr auto proportion(Axis::ZTag, const auto &x,
                                        const auto &y, const auto &z) {
  return proportion(z, sum(x, y, z));
}

[[nodiscard]] constexpr auto cross(const auto &ax, const auto &ay,
                                   const auto &bx, const auto &by) {
  return sub_sat(mul_fixed(ax, by), mul_fixed(ay, bx));
}
[[nodiscard]] constexpr auto cross(Axis::XTag, const auto &, const auto &ay,
                                   const auto &az, const auto &, const auto &by,
                                   const auto &bz) {
  return cross(ay, az, by, bz);
}
[[nodiscard]] constexpr auto cross(Axis::YTag, const auto &ax, const auto &,
                                   const auto &az, const auto &bx, const auto &,
                                   const auto &bz) {
  return cross(az, ax, bz, bx);
}
[[nodiscard]] constexpr auto cross(Axis::ZTag, const auto &ax, const auto &ay,
                                   const auto &, const auto &bx, const auto &by,
                                   const auto &) {
  return cross(ax, ay, bx, by);
}

[[nodiscard]] constexpr auto orient(const auto &ax, const auto &ay,
                                    const auto &bx, const auto &by,
                                    const auto &cx, const auto &cy) {
  return cross(sub_sat(bx, ax), sub_sat(by, ay), sub_sat(cx, ax),
               sub_sat(cy, ay));
}

[[nodiscard]] constexpr auto bary(Axis::XTag, const auto &px, const auto &py,
                                  const auto &ax, const auto &ay,
                                  const auto &bx, const auto &by,
                                  const auto &cx, const auto &cy) {
  return ratio(orient(px, py, bx, by, cx, cy), orient(ax, ay, bx, by, cx, cy));
}
[[nodiscard]] constexpr auto bary(Axis::YTag, const auto &px, const auto &py,
                                  const auto &ax, const auto &ay,
                                  const auto &bx, const auto &by,
                                  const auto &cx, const auto &cy) {
  return ratio(orient(ax, ay, px, py, cx, cy), orient(ax, ay, bx, by, cx, cy));
}
[[nodiscard]] constexpr auto bary(Axis::ZTag, const auto &px, const auto &py,
                                  const auto &ax, const auto &ay,
                                  const auto &bx, const auto &by,
                                  const auto &cx, const auto &cy) {
  return ratio(orient(ax, ay, bx, by, px, py), orient(ax, ay, bx, by, cx, cy));
}

[[nodiscard]] constexpr auto unit(Axis::XTag, const auto &x, const auto &y) {
  return ratio(x, len(x, y));
}
[[nodiscard]] constexpr auto unit(Axis::YTag, const auto &x, const auto &y) {
  return ratio(y, len(x, y));
}
[[nodiscard]] constexpr auto unit(Axis::XTag, const auto &x, const auto &y,
                                  const auto &z) {
  return ratio(x, len(x, y, z));
}
[[nodiscard]] constexpr auto unit(Axis::YTag, const auto &x, const auto &y,
                                  const auto &z) {
  return ratio(y, len(x, y, z));
}
[[nodiscard]] constexpr auto unit(Axis::ZTag, const auto &x, const auto &y,
                                  const auto &z) {
  return ratio(z, len(x, y, z));
}

[[nodiscard]] constexpr auto proj(Axis::XTag, const auto &ax, const auto &ay,
                                  const auto &bx, const auto &by) {
  return mul_fixed(bx, ratio(dot(ax, ay, bx, by), dot(bx, by, bx, by)));
}
[[nodiscard]] constexpr auto proj(Axis::YTag, const auto &ax, const auto &ay,
                                  const auto &bx, const auto &by) {
  return mul_fixed(by, ratio(dot(ax, ay, bx, by), dot(bx, by, bx, by)));
}
[[nodiscard]] constexpr auto proj(Axis::XTag, const auto &ax, const auto &ay,
                                  const auto &az, const auto &bx,
                                  const auto &by, const auto &bz) {
  return mul_fixed(
      bx, ratio(dot(ax, ay, az, bx, by, bz), dot(bx, by, bz, bx, by, bz)));
}
[[nodiscard]] constexpr auto proj(Axis::YTag, const auto &ax, const auto &ay,
                                  const auto &az, const auto &bx,
                                  const auto &by, const auto &bz) {
  return mul_fixed(
      by, ratio(dot(ax, ay, az, bx, by, bz), dot(bx, by, bz, bx, by, bz)));
}
[[nodiscard]] constexpr auto proj(Axis::ZTag, const auto &ax, const auto &ay,
                                  const auto &az, const auto &bx,
                                  const auto &by, const auto &bz) {
  return mul_fixed(
      bz, ratio(dot(ax, ay, az, bx, by, bz), dot(bx, by, bz, bx, by, bz)));
}

[[nodiscard]] constexpr auto reject(Axis::XTag axis, const auto &ax,
                                    const auto &ay, const auto &bx,
                                    const auto &by) {
  return sub_sat(ax, proj(axis, ax, ay, bx, by));
}
[[nodiscard]] constexpr auto reject(Axis::YTag axis, const auto &ax,
                                    const auto &ay, const auto &bx,
                                    const auto &by) {
  return sub_sat(ay, proj(axis, ax, ay, bx, by));
}
[[nodiscard]] constexpr auto reject(Axis::XTag axis, const auto &ax,
                                    const auto &ay, const auto &az,
                                    const auto &bx, const auto &by,
                                    const auto &bz) {
  return sub_sat(ax, proj(axis, ax, ay, az, bx, by, bz));
}
[[nodiscard]] constexpr auto reject(Axis::YTag axis, const auto &ax,
                                    const auto &ay, const auto &az,
                                    const auto &bx, const auto &by,
                                    const auto &bz) {
  return sub_sat(ay, proj(axis, ax, ay, az, bx, by, bz));
}
[[nodiscard]] constexpr auto reject(Axis::ZTag axis, const auto &ax,
                                    const auto &ay, const auto &az,
                                    const auto &bx, const auto &by,
                                    const auto &bz) {
  return sub_sat(az, proj(axis, ax, ay, az, bx, by, bz));
}

[[nodiscard]] constexpr auto reflect(Axis::XTag axis, const auto &ax,
                                     const auto &ay, const auto &bx,
                                     const auto &by) {
  const auto projected = proj(axis, ax, ay, bx, by);
  return sub_sat(add_sat(projected, projected), ax);
}
[[nodiscard]] constexpr auto reflect(Axis::YTag axis, const auto &ax,
                                     const auto &ay, const auto &bx,
                                     const auto &by) {
  const auto projected = proj(axis, ax, ay, bx, by);
  return sub_sat(add_sat(projected, projected), ay);
}
[[nodiscard]] constexpr auto reflect(Axis::XTag axis, const auto &ax,
                                     const auto &ay, const auto &az,
                                     const auto &bx, const auto &by,
                                     const auto &bz) {
  const auto projected = proj(axis, ax, ay, az, bx, by, bz);
  return sub_sat(add_sat(projected, projected), ax);
}
[[nodiscard]] constexpr auto reflect(Axis::YTag axis, const auto &ax,
                                     const auto &ay, const auto &az,
                                     const auto &bx, const auto &by,
                                     const auto &bz) {
  const auto projected = proj(axis, ax, ay, az, bx, by, bz);
  return sub_sat(add_sat(projected, projected), ay);
}
[[nodiscard]] constexpr auto reflect(Axis::ZTag axis, const auto &ax,
                                     const auto &ay, const auto &az,
                                     const auto &bx, const auto &by,
                                     const auto &bz) {
  const auto projected = proj(axis, ax, ay, az, bx, by, bz);
  return sub_sat(add_sat(projected, projected), az);
}

[[nodiscard]] constexpr auto triple(const auto &ax, const auto &ay,
                                    const auto &az, const auto &bx,
                                    const auto &by, const auto &bz,
                                    const auto &cx, const auto &cy,
                                    const auto &cz) {
  return dot(ax, ay, az, cross(Axis::X, bx, by, bz, cx, cy, cz),
             cross(Axis::Y, bx, by, bz, cx, cy, cz),
             cross(Axis::Z, bx, by, bz, cx, cy, cz));
}

[[nodiscard]] constexpr auto line(GeometryOp::ParameterTag, const auto &px,
                                  const auto &py, const auto &ax,
                                  const auto &ay, const auto &bx,
                                  const auto &by) {
  const auto dx = sub_sat(bx, ax);
  const auto dy = sub_sat(by, ay);
  return ratio(dot(sub_sat(px, ax), sub_sat(py, ay), dx, dy),
               len(MetricOp::Squared, dx, dy));
}
[[nodiscard]] constexpr auto line(GeometryOp::DistanceTag, MetricOp::SquaredTag,
                                  const auto &px, const auto &py,
                                  const auto &ax, const auto &ay,
                                  const auto &bx, const auto &by) {
  const auto dx = sub_sat(bx, ax);
  const auto dy = sub_sat(by, ay);
  const auto area = cross(dx, dy, sub_sat(px, ax), sub_sat(py, ay));
  return ratio(mul_fixed(area, area), len(MetricOp::Squared, dx, dy));
}
[[nodiscard]] constexpr auto line(GeometryOp::DistanceTag, const auto &px,
                                  const auto &py, const auto &ax,
                                  const auto &ay, const auto &bx,
                                  const auto &by) {
  return sqrt(
      line(GeometryOp::Distance, MetricOp::Squared, px, py, ax, ay, bx, by));
}
[[nodiscard]] constexpr auto line(GeometryOp::ProjectionTag, Axis::XTag,
                                  const auto &px, const auto &py,
                                  const auto &ax, const auto &ay,
                                  const auto &bx, const auto &by) {
  const auto t = line(GeometryOp::Parameter, px, py, ax, ay, bx, by);
  return add_sat(ax, mul_fixed(sub_sat(bx, ax), t));
}
[[nodiscard]] constexpr auto line(GeometryOp::ProjectionTag, Axis::YTag,
                                  const auto &px, const auto &py,
                                  const auto &ax, const auto &ay,
                                  const auto &bx, const auto &by) {
  const auto t = line(GeometryOp::Parameter, px, py, ax, ay, bx, by);
  return add_sat(ay, mul_fixed(sub_sat(by, ay), t));
}

[[nodiscard]] constexpr auto plane(GeometryOp::ParameterTag, const auto &px,
                                   const auto &py, const auto &pz,
                                   const auto &ax, const auto &ay,
                                   const auto &az, const auto &nx,
                                   const auto &ny, const auto &nz) {
  return ratio(
      dot(sub_sat(px, ax), sub_sat(py, ay), sub_sat(pz, az), nx, ny, nz),
      len(MetricOp::Squared, nx, ny, nz));
}
[[nodiscard]] constexpr auto
plane(GeometryOp::DistanceTag, MetricOp::SquaredTag, const auto &px,
      const auto &py, const auto &pz, const auto &ax, const auto &ay,
      const auto &az, const auto &nx, const auto &ny, const auto &nz) {
  const auto offset =
      dot(sub_sat(px, ax), sub_sat(py, ay), sub_sat(pz, az), nx, ny, nz);
  return ratio(mul_fixed(offset, offset), len(MetricOp::Squared, nx, ny, nz));
}
[[nodiscard]] constexpr auto plane(GeometryOp::DistanceTag, const auto &px,
                                   const auto &py, const auto &pz,
                                   const auto &ax, const auto &ay,
                                   const auto &az, const auto &nx,
                                   const auto &ny, const auto &nz) {
  return sqrt(plane(GeometryOp::Distance, MetricOp::Squared, px, py, pz, ax, ay,
                    az, nx, ny, nz));
}
[[nodiscard]] constexpr auto
plane(GeometryOp::ProjectionTag, Axis::XTag, const auto &px, const auto &py,
      const auto &pz, const auto &ax, const auto &ay, const auto &az,
      const auto &nx, const auto &ny, const auto &nz) {
  const auto t =
      plane(GeometryOp::Parameter, px, py, pz, ax, ay, az, nx, ny, nz);
  return sub_sat(px, mul_fixed(nx, t));
}
[[nodiscard]] constexpr auto
plane(GeometryOp::ProjectionTag, Axis::YTag, const auto &px, const auto &py,
      const auto &pz, const auto &ax, const auto &ay, const auto &az,
      const auto &nx, const auto &ny, const auto &nz) {
  const auto t =
      plane(GeometryOp::Parameter, px, py, pz, ax, ay, az, nx, ny, nz);
  return sub_sat(py, mul_fixed(ny, t));
}
[[nodiscard]] constexpr auto
plane(GeometryOp::ProjectionTag, Axis::ZTag, const auto &px, const auto &py,
      const auto &pz, const auto &ax, const auto &ay, const auto &az,
      const auto &nx, const auto &ny, const auto &nz) {
  const auto t =
      plane(GeometryOp::Parameter, px, py, pz, ax, ay, az, nx, ny, nz);
  return sub_sat(pz, mul_fixed(nz, t));
}

} // namespace rund::compute
