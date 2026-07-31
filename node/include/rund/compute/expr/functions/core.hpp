#pragma once

#include <rund/compute/expr/select.hpp>

#include <type_traits>

namespace rund::compute {

struct FixedOp final {
  struct HalfTag final {};
  struct ThirdTag final {};
  struct QuarterTag final {};

  inline static constexpr HalfTag Half{};
  inline static constexpr ThirdTag Third{};
  inline static constexpr QuarterTag Quarter{};
};

namespace detail {

template <class T> struct FunctionExprValue;
template <class T> struct FunctionExprValue<Expr<T>> final {
  using Type = T;
};
template <class T, class Node>
struct FunctionExprValue<StaticExpr<T, Node>> final {
  using Type = T;
};
template <class E>
using FunctionExprValueT =
    typename FunctionExprValue<std::remove_cvref_t<E>>::Type;

template <class E>
concept FixedExpression = FixedValue<FunctionExprValueT<E>>;

template <class T>
  requires FixedValue<T>
[[nodiscard]] Expr<T> storage(const Expr<T> &value) {
  ExprRef source = ExprAccess::ref(value);
  const FixedFormat format = fixed_storage_format<T>(source.fixed_format);
  return ExprAccess::make<T>(
      quantize_expr(std::move(source), type<T>(), format));
}

template <class T, class Node>
  requires FixedValue<T>
[[nodiscard]] constexpr auto storage(const StaticExpr<T, Node> &value) {
  using Source = StaticExpr<T, Node>;
  return StaticExpr<T, StaticQuantizeLike<T, Source>>{{value}};
}

template <class T>
[[nodiscard]] Expr<T> literal_like(const Expr<T> &anchor, const T value) {
  const ExprRef source = ExprAccess::ref(anchor);
  return ExprAccess::make<T>(
      constant(source.state, type<T>(), static_bits(value),
               fixed_literal_format<T>(source.fixed_format)));
}

template <class T, class Node>
[[nodiscard]] constexpr auto literal_like(const StaticExpr<T, Node> &anchor,
                                          const T value) {
  using Anchor = StaticExpr<T, Node>;
  return StaticExpr<T, StaticLiteralLike<T, Anchor>>{
      StaticLiteralLike<T, Anchor>{anchor, value}};
}

template <class E>
[[nodiscard]] constexpr auto ratio_like(const E &anchor,
                                        const std::int64_t numerator,
                                        const std::int64_t denominator) {
  using T = FunctionExprValueT<E>;
  using Raw = typename T::Raw;
  return literal_like(anchor,
                      T::from_raw(fixed_ratio_raw<Raw, T::fraction_bits>(
                          numerator, denominator, Rounding::NearestEven)));
}

} // namespace detail

template <class E>
  requires detail::FixedExpression<E>
[[nodiscard]] constexpr auto fixed_zero(const E &anchor) {
  using T = detail::FunctionExprValueT<E>;
  return detail::literal_like(anchor, T::zero());
}

template <class E>
  requires detail::FixedExpression<E>
[[nodiscard]] constexpr auto fixed_max(const E &anchor) {
  using T = detail::FunctionExprValueT<E>;
  return detail::literal_like(anchor, T::max());
}

template <class E>
  requires detail::FixedExpression<E>
[[nodiscard]] constexpr auto fixed_one(const E &anchor) {
  return detail::ratio_like(anchor, 1, 1);
}

template <class E>
  requires detail::FixedExpression<E>
[[nodiscard]] constexpr auto fixed(FixedOp::HalfTag, const E &anchor) {
  return detail::ratio_like(anchor, 1, 2);
}

template <class E>
  requires detail::FixedExpression<E>
[[nodiscard]] constexpr auto fixed(FixedOp::ThirdTag, const E &anchor) {
  return detail::ratio_like(anchor, 1, 3);
}

template <class E>
  requires detail::FixedExpression<E>
[[nodiscard]] constexpr auto fixed(FixedOp::QuarterTag, const E &anchor) {
  return detail::ratio_like(anchor, 1, 4);
}

template <class Left, class Right>
  requires(detail::FixedExpression<Left> && detail::FixedExpression<Right> &&
           requires(const Left &lhs, const Right &rhs) { lhs / rhs; })
[[nodiscard]] constexpr auto div_fixed(const Left &left, const Right &right) {
  return left / right;
}

[[nodiscard]] constexpr auto neg(const auto &value) { return -value; }

[[nodiscard]] constexpr auto eq(const auto &left, const auto &right) {
  return left == right;
}
[[nodiscard]] constexpr auto ne(const auto &left, const auto &right) {
  return left != right;
}
[[nodiscard]] constexpr auto lt(const auto &left, const auto &right) {
  return left < right;
}
[[nodiscard]] constexpr auto le(const auto &left, const auto &right) {
  return left <= right;
}
[[nodiscard]] constexpr auto gt(const auto &left, const auto &right) {
  return left > right;
}
[[nodiscard]] constexpr auto ge(const auto &left, const auto &right) {
  return left >= right;
}

[[nodiscard]] constexpr auto predicate_not(const auto &value) { return !value; }
[[nodiscard]] constexpr auto predicate_and(const auto &left,
                                           const auto &right) {
  return left && right;
}
[[nodiscard]] constexpr auto predicate_or(const auto &left, const auto &right) {
  return left || right;
}

[[nodiscard]] constexpr auto bit_and(const auto &left, const auto &right) {
  return left & right;
}
[[nodiscard]] constexpr auto bit_or(const auto &left, const auto &right) {
  return left | right;
}
[[nodiscard]] constexpr auto bit_xor(const auto &left, const auto &right) {
  return left ^ right;
}

[[nodiscard]] constexpr auto saturate(const auto &value) {
  return clamp(value, fixed_zero(value), fixed_one(value));
}

[[nodiscard]] constexpr auto step(const auto &edge, const auto &value) {
  return select(value < edge, fixed_zero(value), fixed_one(value));
}

[[nodiscard]] constexpr auto absdiff(const auto &left, const auto &right) {
  return abs(sub_sat(left, right));
}

[[nodiscard]] constexpr auto midrange(const auto &left, const auto &right) {
  return mean(left, right);
}

[[nodiscard]] constexpr auto median(const auto &first, const auto &second,
                                    const auto &third) {
  return max(min(first, second), min(max(first, second), third));
}

[[nodiscard]] constexpr auto spread(const auto &first, const auto &second,
                                    const auto &third) {
  return sub_sat(max(max(first, second), third),
                 min(min(first, second), third));
}

[[nodiscard]] constexpr auto clamp_range(const auto &value, const auto &first,
                                         const auto &second) {
  return clamp(value, min(first, second), max(first, second));
}

[[nodiscard]] constexpr auto in_range(const auto &value, const auto &low,
                                      const auto &high) {
  return (value >= low) && (value <= high);
}

[[nodiscard]] constexpr auto out_range(const auto &value, const auto &low,
                                       const auto &high) {
  return !in_range(value, low, high);
}

[[nodiscard]] constexpr auto bandpass(const auto &value, const auto &first,
                                      const auto &second) {
  const auto low = min(first, second);
  const auto high = max(first, second);
  return select(in_range(value, low, high), value, fixed_zero(value));
}

[[nodiscard]] constexpr auto bandstop(const auto &value, const auto &first,
                                      const auto &second) {
  const auto low = min(first, second);
  const auto high = max(first, second);
  return select(in_range(value, low, high), fixed_zero(value), value);
}

[[nodiscard]] constexpr auto is_zero(const auto &value) {
  return value == fixed_zero(value);
}
[[nodiscard]] constexpr auto nonzero(const auto &value) {
  return value != fixed_zero(value);
}
[[nodiscard]] constexpr auto is_neg(const auto &value) {
  return value < fixed_zero(value);
}
[[nodiscard]] constexpr auto is_pos(const auto &value) {
  return value > fixed_zero(value);
}
[[nodiscard]] constexpr auto is_nonneg(const auto &value) {
  return value >= fixed_zero(value);
}
[[nodiscard]] constexpr auto is_nonpos(const auto &value) {
  return value <= fixed_zero(value);
}

[[nodiscard]] constexpr auto all(const auto &a, const auto &b) {
  return a && b;
}
[[nodiscard]] constexpr auto all(const auto &a, const auto &b, const auto &c) {
  return (a && b) && c;
}
[[nodiscard]] constexpr auto all(const auto &a, const auto &b, const auto &c,
                                 const auto &d) {
  return ((a && b) && c) && d;
}
[[nodiscard]] constexpr auto any(const auto &a, const auto &b) {
  return a || b;
}
[[nodiscard]] constexpr auto any(const auto &a, const auto &b, const auto &c) {
  return (a || b) || c;
}
[[nodiscard]] constexpr auto any(const auto &a, const auto &b, const auto &c,
                                 const auto &d) {
  return ((a || b) || c) || d;
}

[[nodiscard]] constexpr auto keep_if(const auto &condition, const auto &value) {
  return select(condition, value, fixed_zero(value));
}
[[nodiscard]] constexpr auto zero_if(const auto &condition, const auto &value) {
  return select(condition, fixed_zero(value), value);
}

[[nodiscard]] constexpr auto near(const auto &value, const auto &target,
                                  const auto &tolerance) {
  return absdiff(value, target) <= abs(tolerance);
}
[[nodiscard]] constexpr auto near(const auto &value, const auto &tolerance) {
  return near(value, fixed_zero(value), tolerance);
}
[[nodiscard]] constexpr auto deadzone(const auto &value,
                                      const auto &tolerance) {
  return select(near(value, tolerance), fixed_zero(value), value);
}
[[nodiscard]] constexpr auto snap(const auto &value, const auto &target,
                                  const auto &tolerance) {
  return select(near(value, target, tolerance), target, value);
}

[[nodiscard]] constexpr auto clip(const auto &value, const auto &bound) {
  const auto magnitude = detail::storage(abs(bound));
  return clamp(value, -magnitude, magnitude);
}
[[nodiscard]] constexpr auto positive_part(const auto &value) {
  return max(value, fixed_zero(value));
}
[[nodiscard]] constexpr auto negative_part(const auto &value) {
  return min(value, fixed_zero(value));
}

[[nodiscard]] constexpr auto lerp(const auto &left, const auto &right,
                                  const auto &amount) {
  const auto t = saturate(amount);
  return add_sat(mul_fixed(left, sub_sat(fixed_one(amount), t)),
                 mul_fixed(right, t));
}

[[nodiscard]] constexpr auto lerp(const auto &x00, const auto &x10,
                                  const auto &x01, const auto &x11,
                                  const auto &tx, const auto &ty) {
  return lerp(lerp(x00, x10, tx), lerp(x01, x11, tx), ty);
}

[[nodiscard]] constexpr auto
lerp(const auto &x000, const auto &x100, const auto &x010, const auto &x110,
     const auto &x001, const auto &x101, const auto &x011, const auto &x111,
     const auto &tx, const auto &ty, const auto &tz) {
  return lerp(lerp(x000, x100, x010, x110, tx, ty),
              lerp(x001, x101, x011, x111, tx, ty), tz);
}

[[nodiscard]] constexpr auto unlerp(const auto &low, const auto &high,
                                    const auto &value) {
  return select(high <= low, fixed_zero(value),
                saturate(div_fixed(sub_sat(value, low), sub_sat(high, low))));
}

[[nodiscard]] constexpr auto remap(const auto &input_low,
                                   const auto &input_high,
                                   const auto &output_low,
                                   const auto &output_high, const auto &value) {
  return lerp(output_low, output_high, unlerp(input_low, input_high, value));
}
[[nodiscard]] constexpr auto bezier(const auto &a, const auto &b, const auto &c,
                                    const auto &amount) {
  return lerp(lerp(a, b, amount), lerp(b, c, amount), amount);
}
[[nodiscard]] constexpr auto bezier(const auto &a, const auto &b, const auto &c,
                                    const auto &d, const auto &amount) {
  return lerp(bezier(a, b, c, amount), bezier(b, c, d, amount), amount);
}

[[nodiscard]] constexpr auto fade(const auto &amount) {
  const auto t = saturate(amount);
  const auto t2 = mul_fixed(t, t);
  const auto inverse = sub_sat(fixed_one(t), t);
  const auto bump = mul_fixed(t2, inverse);
  return add_sat(t2, add_sat(bump, bump));
}

[[nodiscard]] constexpr auto smoothstep(const auto &edge0, const auto &edge1,
                                        const auto &value) {
  return fade(unlerp(edge0, edge1, value));
}

} // namespace rund::compute
