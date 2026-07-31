#pragma once

#include <rund/compute/expr/static.hpp>

namespace rund::compute {
namespace detail {
template <ExprOp Op, class T, class Node>
[[nodiscard]] constexpr auto static_unary(const StaticExpr<T, Node> &value) {
  return StaticExpr<T, StaticUnary<Op, StaticExpr<T, Node>>>{{value}};
}
template <ExprOp Op, class T, class Left, class Right>
[[nodiscard]] constexpr auto static_binary(const StaticExpr<T, Left> &left,
                                           const StaticExpr<T, Right> &right) {
  return StaticExpr<
      T, StaticBinary<Op, StaticExpr<T, Left>, StaticExpr<T, Right>>>{
      {left, right}};
}
template <ExprOp Op, class T, class First, class Second, class Third>
[[nodiscard]] constexpr auto static_ternary(const StaticExpr<T, First> &first,
                                            const StaticExpr<T, Second> &second,
                                            const StaticExpr<T, Third> &third) {
  return StaticExpr<T,
                    StaticTernary<Op, StaticExpr<T, First>,
                                  StaticExpr<T, Second>, StaticExpr<T, Third>>>{
      {first, second, third}};
}
template <class T>
concept SignedExprValue = std::signed_integral<T> || FixedValue<T>;
template <class T>
concept UnsignedStorageExprValue = std::unsigned_integral<T> || FixedValue<T>;
template <class T>
concept StoredExprValue = std::integral<T> || FixedValue<T>;
} // namespace detail

#define RUND_COMPUTE_STATIC_UNARY(name, operation, constraint)                 \
  template <class T, class Node>                                               \
    requires constraint<T>                                                     \
  [[nodiscard]] constexpr auto name(                                           \
      const detail::StaticExpr<T, Node> &value) {                              \
    return detail::static_unary<detail::ExprOp::operation>(value);             \
  }
RUND_COMPUTE_STATIC_UNARY(abs, Abs, detail::SignedExprValue)
RUND_COMPUTE_STATIC_UNARY(abs_magnitude, AbsMagnitude, detail::SignedExprValue)
RUND_COMPUTE_STATIC_UNARY(sign, Sign, detail::SignedExprValue)
RUND_COMPUTE_STATIC_UNARY(bit_not, BitNot, detail::StoredExprValue)
RUND_COMPUTE_STATIC_UNARY(neg_positive_fixed, NegPositiveFixed,
                          detail::FixedValue)
RUND_COMPUTE_STATIC_UNARY(recip, Reciprocal, detail::FixedValue)
RUND_COMPUTE_STATIC_UNARY(sqrt, Sqrt, detail::FixedValue)
RUND_COMPUTE_STATIC_UNARY(rsqrt, Rsqrt, detail::FixedValue)
RUND_COMPUTE_STATIC_UNARY(sin, Sin, detail::FixedValue)
RUND_COMPUTE_STATIC_UNARY(cos, Cos, detail::FixedValue)
RUND_COMPUTE_STATIC_UNARY(tan, Tan, detail::FixedValue)
RUND_COMPUTE_STATIC_UNARY(exp, Exp, detail::FixedValue)
RUND_COMPUTE_STATIC_UNARY(log, Log, detail::FixedValue)
#undef RUND_COMPUTE_STATIC_UNARY

#define RUND_COMPUTE_STATIC_BINARY_FUNCTION(name, operation, constraint)       \
  template <class T, class Left, class Right>                                  \
    requires constraint<T>                                                     \
  [[nodiscard]] constexpr auto name(                                           \
      const detail::StaticExpr<T, Left> &left,                                 \
      const detail::StaticExpr<T, Right> &right) {                             \
    return detail::static_binary<detail::ExprOp::operation>(left, right);      \
  }
RUND_COMPUTE_STATIC_BINARY_FUNCTION(add_sat, AddSat, detail::SignedExprValue)
RUND_COMPUTE_STATIC_BINARY_FUNCTION(add_sat_unsigned, AddSatUnsigned,
                                    detail::UnsignedStorageExprValue)
RUND_COMPUTE_STATIC_BINARY_FUNCTION(sub_sat, SubSat, detail::SignedExprValue)
RUND_COMPUTE_STATIC_BINARY_FUNCTION(mul_wrap, MultiplyWrap,
                                    detail::StoredExprValue)
RUND_COMPUTE_STATIC_BINARY_FUNCTION(mul_fixed, MulFixed, detail::FixedValue)
RUND_COMPUTE_STATIC_BINARY_FUNCTION(mul_fixed_scaled, MulFixedScaled,
                                    detail::FixedValue)
RUND_COMPUTE_STATIC_BINARY_FUNCTION(mul_unsigned_fixed, MulUnsignedFixed,
                                    detail::FixedValue)
RUND_COMPUTE_STATIC_BINARY_FUNCTION(atan2, Atan2, detail::FixedValue)
#undef RUND_COMPUTE_STATIC_BINARY_FUNCTION

template <class T, class First, class Second, class Third>
  requires detail::FixedValue<T>
[[nodiscard]] constexpr auto
mul_add_fixed(const detail::StaticExpr<T, First> &left,
              const detail::StaticExpr<T, Second> &right,
              const detail::StaticExpr<T, Third> &addend) {
  return detail::static_ternary<detail::ExprOp::MulAddFixed>(left, right,
                                                             addend);
}

template <class T, class Left, class Right>
  requires detail::FixedValue<T>
[[nodiscard]] constexpr auto pow(const detail::StaticExpr<T, Left> &base,
                                 const detail::StaticExpr<T, Right> &exponent) {
  return exp(quantize<T, Rounding::NearestEven, Overflow::Saturate,
                      Approximation::Deterministic>(exponent * log(base)));
}

template <std::uint32_t Amount, class T, class Node>
  requires detail::StoredExprValue<T>
[[nodiscard]] constexpr auto shl(const detail::StaticExpr<T, Node> &value) {
  static_assert(Amount < sizeof(T) * 8u, "shift amount exceeds value width");
  return detail::StaticExpr<
      T, detail::StaticShift<detail::ExprOp::ShiftLeft, Amount,
                             detail::StaticExpr<T, Node>>>{{value}};
}
template <std::uint32_t Amount, class T, class Node>
  requires detail::StoredExprValue<T>
[[nodiscard]] constexpr auto
shr_logical(const detail::StaticExpr<T, Node> &value) {
  static_assert(Amount < sizeof(T) * 8u, "shift amount exceeds value width");
  return detail::StaticExpr<
      T, detail::StaticShift<detail::ExprOp::ShiftRightLogical, Amount,
                             detail::StaticExpr<T, Node>>>{{value}};
}
template <std::uint32_t Amount, class T, class Node>
  requires detail::SignedExprValue<T>
[[nodiscard]] constexpr auto
shr_arithmetic(const detail::StaticExpr<T, Node> &value) {
  static_assert(Amount < sizeof(T) * 8u, "shift amount exceeds value width");
  return detail::StaticExpr<
      T, detail::StaticShift<detail::ExprOp::ShiftRightArithmetic, Amount,
                             detail::StaticExpr<T, Node>>>{{value}};
}
} // namespace rund::compute
