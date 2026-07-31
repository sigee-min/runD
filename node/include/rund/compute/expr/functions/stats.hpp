#pragma once

#include <rund/compute/expr/functions/linear.hpp>

namespace rund::compute {

struct MeanOp final {
  struct AbsTag final {};
  struct SquaredTag final {};

  inline static constexpr AbsTag Abs{};
  inline static constexpr SquaredTag Squared{};
};
struct CenteredOp final {
  struct AbsTag final {};
  struct SquaredTag final {};
  struct CubicTag final {};
  struct QuarticTag final {};

  inline static constexpr AbsTag Abs{};
  inline static constexpr SquaredTag Squared{};
  inline static constexpr CubicTag Cubic{};
  inline static constexpr QuarticTag Quartic{};
};
struct SumOp final {
  struct AbsTag final {};
  struct SquaredTag final {};

  inline static constexpr AbsTag Abs{};
  inline static constexpr SquaredTag Squared{};
};
struct DifferenceOrder final {
  struct SecondTag final {};
  struct ThirdTag final {};

  inline static constexpr SecondTag Second{};
  inline static constexpr ThirdTag Third{};
};
struct StandardizedOp final {
  struct CubicTag final {};
  struct QuarticTag final {};

  inline static constexpr CubicTag Cubic{};
  inline static constexpr QuarticTag Quartic{};
};

[[nodiscard]] constexpr auto mean(const auto &left, const auto &right) {
  const auto half = fixed(FixedOp::Half, left);
  return add_sat(mul_fixed(left, half), mul_fixed(right, half));
}
[[nodiscard]] constexpr auto mean(const auto &a, const auto &b, const auto &c) {
  const auto third = fixed(FixedOp::Third, a);
  return add_sat(add_sat(mul_fixed(a, third), mul_fixed(b, third)),
                 mul_fixed(c, third));
}
[[nodiscard]] constexpr auto mean(const auto &a, const auto &b, const auto &c,
                                  const auto &d) {
  const auto quarter = fixed(FixedOp::Quarter, a);
  return add_sat(add_sat(mul_fixed(a, quarter), mul_fixed(b, quarter)),
                 add_sat(mul_fixed(c, quarter), mul_fixed(d, quarter)));
}

[[nodiscard]] constexpr auto mean(MeanOp::AbsTag, const auto &a,
                                  const auto &b) {
  return mean(detail::storage(abs(a)), detail::storage(abs(b)));
}
[[nodiscard]] constexpr auto mean(MeanOp::AbsTag, const auto &a, const auto &b,
                                  const auto &c) {
  return mean(detail::storage(abs(a)), detail::storage(abs(b)),
              detail::storage(abs(c)));
}
[[nodiscard]] constexpr auto mean(MeanOp::AbsTag, const auto &a, const auto &b,
                                  const auto &c, const auto &d) {
  return mean(detail::storage(abs(a)), detail::storage(abs(b)),
              detail::storage(abs(c)), detail::storage(abs(d)));
}
[[nodiscard]] constexpr auto mean(MeanOp::SquaredTag, const auto &a,
                                  const auto &b) {
  return mean(mul_fixed(a, a), mul_fixed(b, b));
}
[[nodiscard]] constexpr auto mean(MeanOp::SquaredTag, const auto &a,
                                  const auto &b, const auto &c) {
  return mean(mul_fixed(a, a), mul_fixed(b, b), mul_fixed(c, c));
}
[[nodiscard]] constexpr auto mean(MeanOp::SquaredTag, const auto &a,
                                  const auto &b, const auto &c, const auto &d) {
  return mean(mul_fixed(a, a), mul_fixed(b, b), mul_fixed(c, c),
              mul_fixed(d, d));
}

[[nodiscard]] constexpr auto centered(const auto &value, const auto &center) {
  return sub_sat(value, center);
}
[[nodiscard]] constexpr auto centered(CenteredOp::AbsTag, const auto &value,
                                      const auto &center) {
  return abs(centered(value, center));
}
[[nodiscard]] constexpr auto centered(CenteredOp::SquaredTag, const auto &value,
                                      const auto &center) {
  const auto delta = centered(value, center);
  return mul_fixed(delta, delta);
}
[[nodiscard]] constexpr auto centered(CenteredOp::CubicTag, const auto &value,
                                      const auto &center) {
  const auto delta = centered(value, center);
  return mul_fixed(mul_fixed(delta, delta), delta);
}
[[nodiscard]] constexpr auto centered(CenteredOp::QuarticTag, const auto &value,
                                      const auto &center) {
  const auto square = centered(CenteredOp::Squared, value, center);
  return mul_fixed(square, square);
}

[[nodiscard]] constexpr auto sum(const auto &a, const auto &b)
  requires detail::FixedExpression<decltype(a)>
{
  return add_sat(a, b);
}
[[nodiscard]] constexpr auto sum(const auto &a, const auto &b, const auto &c)
  requires detail::FixedExpression<decltype(a)>
{
  return add_sat(sum(a, b), c);
}
[[nodiscard]] constexpr auto sum(const auto &a, const auto &b, const auto &c,
                                 const auto &d)
  requires detail::FixedExpression<decltype(a)>
{
  return add_sat(sum(a, b), sum(c, d));
}
template <class... Rest>
[[nodiscard]] constexpr auto sum(SumOp::AbsTag, const auto &a, const auto &b,
                                 const Rest &...rest) {
  return sum(detail::storage(abs(a)), detail::storage(abs(b)),
             detail::storage(abs(rest))...);
}
template <class... Rest>
[[nodiscard]] constexpr auto sum(SumOp::SquaredTag, const auto &a,
                                 const auto &b, const Rest &...rest) {
  return sum(mul_fixed(a, a), mul_fixed(b, b), mul_fixed(rest, rest)...);
}

[[nodiscard]] constexpr auto diff(const auto &from, const auto &to) {
  return sub_sat(to, from);
}
[[nodiscard]] constexpr auto diff(const auto &previous, const auto &center,
                                  const auto &next) {
  return mean(diff(previous, center), diff(center, next));
}
[[nodiscard]] constexpr auto diff(DifferenceOrder::SecondTag,
                                  const auto &previous, const auto &center,
                                  const auto &next) {
  return sub_sat(diff(center, next), diff(previous, center));
}
[[nodiscard]] constexpr auto diff(DifferenceOrder::ThirdTag, const auto &a,
                                  const auto &b, const auto &c, const auto &d) {
  return sub_sat(diff(DifferenceOrder::Second, b, c, d),
                 diff(DifferenceOrder::Second, a, b, c));
}

template <class... Rest>
[[nodiscard]] constexpr auto var(const auto &a, const auto &b,
                                 const Rest &...rest) {
  const auto center = mean(a, b, rest...);
  return mean(centered(CenteredOp::Squared, a, center),
              centered(CenteredOp::Squared, b, center),
              centered(CenteredOp::Squared, rest, center)...);
}
template <class... Rest>
[[nodiscard]] constexpr auto rms(const auto &a, const auto &b,
                                 const Rest &...rest) {
  return sqrt(mean(MeanOp::Squared, a, b, rest...));
}

[[nodiscard]] constexpr auto ratio(const auto &numerator,
                                   const auto &denominator) {
  return select(denominator == fixed_zero(denominator), fixed_zero(numerator),
                div_fixed(numerator, denominator));
}
[[nodiscard]] constexpr auto zscore(const auto &value, const auto &center,
                                    const auto &scale) {
  return ratio(centered(value, center), scale);
}
[[nodiscard]] constexpr auto standardized(StandardizedOp::CubicTag,
                                          const auto &value, const auto &center,
                                          const auto &scale) {
  const auto score = zscore(value, center, scale);
  return mul_fixed(mul_fixed(score, score), score);
}
[[nodiscard]] constexpr auto standardized(StandardizedOp::QuarticTag,
                                          const auto &value, const auto &center,
                                          const auto &scale) {
  const auto score = zscore(value, center, scale);
  const auto square = mul_fixed(score, score);
  return mul_fixed(square, square);
}

[[nodiscard]] constexpr auto standard_mean(auto op, const auto &a,
                                           const auto &b) {
  const auto center = mean(a, b);
  const auto scale = sqrt(var(a, b));
  return mean(standardized(op, a, center, scale),
              standardized(op, b, center, scale));
}
[[nodiscard]] constexpr auto standard_mean(auto op, const auto &a,
                                           const auto &b, const auto &c) {
  const auto center = mean(a, b, c);
  const auto scale = sqrt(var(a, b, c));
  return mean(standardized(op, a, center, scale),
              standardized(op, b, center, scale),
              standardized(op, c, center, scale));
}
[[nodiscard]] constexpr auto standard_mean(auto op, const auto &a,
                                           const auto &b, const auto &c,
                                           const auto &d) {
  const auto center = mean(a, b, c, d);
  const auto scale = sqrt(var(a, b, c, d));
  return mean(
      standardized(op, a, center, scale), standardized(op, b, center, scale),
      standardized(op, c, center, scale), standardized(op, d, center, scale));
}
#define RUND_COMPUTE_STANDARD_MEAN(op_type)                                    \
  [[nodiscard]] constexpr auto mean(op_type op, const auto &a,                 \
                                    const auto &b) {                           \
    return standard_mean(op, a, b);                                            \
  }                                                                            \
  [[nodiscard]] constexpr auto mean(op_type op, const auto &a, const auto &b,  \
                                    const auto &c) {                           \
    return standard_mean(op, a, b, c);                                         \
  }                                                                            \
  [[nodiscard]] constexpr auto mean(op_type op, const auto &a, const auto &b,  \
                                    const auto &c, const auto &d) {            \
    return standard_mean(op, a, b, c, d);                                      \
  }
RUND_COMPUTE_STANDARD_MEAN(StandardizedOp::CubicTag)
RUND_COMPUTE_STANDARD_MEAN(StandardizedOp::QuarticTag)
#undef RUND_COMPUTE_STANDARD_MEAN

[[nodiscard]] constexpr auto cov(const auto &x0, const auto &x1, const auto &y0,
                                 const auto &y1) {
  const auto mx = mean(x0, x1);
  const auto my = mean(y0, y1);
  return mean(mul_fixed(centered(x0, mx), centered(y0, my)),
              mul_fixed(centered(x1, mx), centered(y1, my)));
}
[[nodiscard]] constexpr auto cov(const auto &x0, const auto &x1, const auto &x2,
                                 const auto &y0, const auto &y1,
                                 const auto &y2) {
  const auto mx = mean(x0, x1, x2);
  const auto my = mean(y0, y1, y2);
  return mean(mul_fixed(centered(x0, mx), centered(y0, my)),
              mul_fixed(centered(x1, mx), centered(y1, my)),
              mul_fixed(centered(x2, mx), centered(y2, my)));
}
[[nodiscard]] constexpr auto cov(const auto &x0, const auto &x1, const auto &x2,
                                 const auto &x3, const auto &y0, const auto &y1,
                                 const auto &y2, const auto &y3) {
  const auto mx = mean(x0, x1, x2, x3);
  const auto my = mean(y0, y1, y2, y3);
  return mean(mul_fixed(centered(x0, mx), centered(y0, my)),
              mul_fixed(centered(x1, mx), centered(y1, my)),
              mul_fixed(centered(x2, mx), centered(y2, my)),
              mul_fixed(centered(x3, mx), centered(y3, my)));
}
[[nodiscard]] constexpr auto corr(const auto &x0, const auto &x1,
                                  const auto &y0, const auto &y1) {
  const auto denominator = sqrt(mul_fixed(var(x0, x1), var(y0, y1)));
  return ratio(cov(x0, x1, y0, y1), denominator);
}
[[nodiscard]] constexpr auto corr(const auto &x0, const auto &x1,
                                  const auto &x2, const auto &y0,
                                  const auto &y1, const auto &y2) {
  const auto denominator = sqrt(mul_fixed(var(x0, x1, x2), var(y0, y1, y2)));
  return ratio(cov(x0, x1, x2, y0, y1, y2), denominator);
}
[[nodiscard]] constexpr auto corr(const auto &x0, const auto &x1,
                                  const auto &x2, const auto &x3,
                                  const auto &y0, const auto &y1,
                                  const auto &y2, const auto &y3) {
  const auto denominator =
      sqrt(mul_fixed(var(x0, x1, x2, x3), var(y0, y1, y2, y3)));
  return ratio(cov(x0, x1, x2, x3, y0, y1, y2, y3), denominator);
}

} // namespace rund::compute
