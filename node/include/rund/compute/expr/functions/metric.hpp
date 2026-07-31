#pragma once

#include <rund/compute/expr/functions/stats.hpp>

namespace rund::compute {

struct MetricOp final {
  struct SquaredTag final {};

  inline static constexpr SquaredTag Squared{};
};
struct AngleOp final {
  struct CosineTag final {};

  inline static constexpr CosineTag Cosine{};
};
struct Norm final {
  struct L1Tag final {};
  struct LInfTag final {};
  inline static constexpr L1Tag L1{};
  inline static constexpr LInfTag LInf{};
};

[[nodiscard]] constexpr auto len(MetricOp::SquaredTag, const auto &x,
                                 const auto &y) {
  return dot(x, y, x, y);
}
[[nodiscard]] constexpr auto len(MetricOp::SquaredTag, const auto &x,
                                 const auto &y, const auto &z) {
  return dot(x, y, z, x, y, z);
}
[[nodiscard]] constexpr auto len(const auto &x, const auto &y) {
  return sqrt(len(MetricOp::Squared, x, y));
}
[[nodiscard]] constexpr auto len(const auto &x, const auto &y, const auto &z) {
  return sqrt(len(MetricOp::Squared, x, y, z));
}
[[nodiscard]] constexpr auto len(Norm::L1Tag, const auto &x, const auto &y) {
  return add_sat(detail::storage(abs(x)), detail::storage(abs(y)));
}
[[nodiscard]] constexpr auto len(Norm::L1Tag norm, const auto &x, const auto &y,
                                 const auto &z) {
  return add_sat(len(norm, x, y), detail::storage(abs(z)));
}
[[nodiscard]] constexpr auto len(Norm::LInfTag, const auto &x, const auto &y) {
  return max(abs(x), abs(y));
}
[[nodiscard]] constexpr auto len(Norm::LInfTag norm, const auto &x,
                                 const auto &y, const auto &z) {
  return max(len(norm, x, y), abs(z));
}

[[nodiscard]] constexpr auto dist(MetricOp::SquaredTag op, const auto &ax,
                                  const auto &ay, const auto &bx,
                                  const auto &by) {
  return len(op, sub_sat(ax, bx), sub_sat(ay, by));
}
[[nodiscard]] constexpr auto dist(MetricOp::SquaredTag op, const auto &ax,
                                  const auto &ay, const auto &az,
                                  const auto &bx, const auto &by,
                                  const auto &bz) {
  return len(op, sub_sat(ax, bx), sub_sat(ay, by), sub_sat(az, bz));
}
[[nodiscard]] constexpr auto dist(const auto &ax, const auto &ay,
                                  const auto &bx, const auto &by) {
  return sqrt(dist(MetricOp::Squared, ax, ay, bx, by));
}
[[nodiscard]] constexpr auto dist(const auto &ax, const auto &ay,
                                  const auto &az, const auto &bx,
                                  const auto &by, const auto &bz) {
  return sqrt(dist(MetricOp::Squared, ax, ay, az, bx, by, bz));
}
[[nodiscard]] constexpr auto dist(Norm::L1Tag norm, const auto &ax,
                                  const auto &ay, const auto &bx,
                                  const auto &by) {
  return len(norm, sub_sat(ax, bx), sub_sat(ay, by));
}
[[nodiscard]] constexpr auto dist(Norm::L1Tag norm, const auto &ax,
                                  const auto &ay, const auto &az,
                                  const auto &bx, const auto &by,
                                  const auto &bz) {
  return len(norm, sub_sat(ax, bx), sub_sat(ay, by), sub_sat(az, bz));
}
[[nodiscard]] constexpr auto dist(Norm::LInfTag norm, const auto &ax,
                                  const auto &ay, const auto &bx,
                                  const auto &by) {
  return len(norm, sub_sat(ax, bx), sub_sat(ay, by));
}
[[nodiscard]] constexpr auto dist(Norm::LInfTag norm, const auto &ax,
                                  const auto &ay, const auto &az,
                                  const auto &bx, const auto &by,
                                  const auto &bz) {
  return len(norm, sub_sat(ax, bx), sub_sat(ay, by), sub_sat(az, bz));
}

[[nodiscard]] constexpr auto angle(AngleOp::CosineTag, const auto &ax,
                                   const auto &ay, const auto &bx,
                                   const auto &by) {
  const auto denominator = sqrt(mul_fixed(len(MetricOp::Squared, ax, ay),
                                          len(MetricOp::Squared, bx, by)));
  return select(denominator == fixed_zero(denominator), fixed_zero(ax),
                div_fixed(dot(ax, ay, bx, by), denominator));
}
[[nodiscard]] constexpr auto angle(AngleOp::CosineTag, const auto &ax,
                                   const auto &ay, const auto &az,
                                   const auto &bx, const auto &by,
                                   const auto &bz) {
  const auto denominator = sqrt(mul_fixed(len(MetricOp::Squared, ax, ay, az),
                                          len(MetricOp::Squared, bx, by, bz)));
  return select(denominator == fixed_zero(denominator), fixed_zero(ax),
                div_fixed(dot(ax, ay, az, bx, by, bz), denominator));
}

} // namespace rund::compute
