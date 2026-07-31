#pragma once

#include <rund/compute/expr/record.hpp>

namespace rund::compute {
template <class Target, Rounding Round, Overflow OverflowMode,
          Approximation ApproximationMode, class Source>
  requires(detail::FixedValue<Target> && detail::FixedValue<Source>)
[[nodiscard]] Expr<Target> quantize(const Expr<Source> &value) {
  return detail::ExprAccess::make<Target>(detail::quantize_expr(
      detail::ExprAccess::ref(value), detail::type<Target>(),
      detail::fixed_format<Target>(Round, OverflowMode, ApproximationMode)));
}

#define RUND_COMPUTE_EXPR_UNARY(name, operation, constraint)                   \
  template <class T>                                                           \
    requires constraint<T>                                                     \
  [[nodiscard]] Expr<T> name(const Expr<T> &value) {                           \
    return detail::ExprAccess::make<T>(detail::unary(                          \
        detail::ExprOp::operation, detail::ExprAccess::ref(value)));           \
  }
RUND_COMPUTE_EXPR_UNARY(abs, Abs, detail::SignedExprValue)
RUND_COMPUTE_EXPR_UNARY(abs_magnitude, AbsMagnitude, detail::SignedExprValue)
RUND_COMPUTE_EXPR_UNARY(sign, Sign, detail::SignedExprValue)
RUND_COMPUTE_EXPR_UNARY(bit_not, BitNot, detail::StoredExprValue)
RUND_COMPUTE_EXPR_UNARY(neg_positive_fixed, NegPositiveFixed,
                        detail::FixedValue)
RUND_COMPUTE_EXPR_UNARY(recip, Reciprocal, detail::FixedValue)
RUND_COMPUTE_EXPR_UNARY(sqrt, Sqrt, detail::FixedValue)
RUND_COMPUTE_EXPR_UNARY(rsqrt, Rsqrt, detail::FixedValue)
RUND_COMPUTE_EXPR_UNARY(sin, Sin, detail::FixedValue)
RUND_COMPUTE_EXPR_UNARY(cos, Cos, detail::FixedValue)
RUND_COMPUTE_EXPR_UNARY(tan, Tan, detail::FixedValue)
RUND_COMPUTE_EXPR_UNARY(exp, Exp, detail::FixedValue)
RUND_COMPUTE_EXPR_UNARY(log, Log, detail::FixedValue)
#undef RUND_COMPUTE_EXPR_UNARY

#define RUND_COMPUTE_EXPR_BINARY(name, operation, constraint)                  \
  template <class T>                                                           \
    requires constraint<T>                                                     \
  [[nodiscard]] Expr<T> name(const Expr<T> &left, const Expr<T> &right) {      \
    return detail::ExprAccess::make<T>(detail::binary(                         \
        detail::ExprOp::operation, detail::ExprAccess::ref(left),              \
        detail::ExprAccess::ref(right)));                                      \
  }
RUND_COMPUTE_EXPR_BINARY(add_sat, AddSat, detail::SignedExprValue)
RUND_COMPUTE_EXPR_BINARY(add_sat_unsigned, AddSatUnsigned,
                         detail::UnsignedStorageExprValue)
RUND_COMPUTE_EXPR_BINARY(sub_sat, SubSat, detail::SignedExprValue)
RUND_COMPUTE_EXPR_BINARY(mul_wrap, MultiplyWrap, detail::StoredExprValue)
RUND_COMPUTE_EXPR_BINARY(mul_fixed, MulFixed, detail::FixedValue)
RUND_COMPUTE_EXPR_BINARY(mul_fixed_scaled, MulFixedScaled, detail::FixedValue)
RUND_COMPUTE_EXPR_BINARY(mul_unsigned_fixed, MulUnsignedFixed,
                         detail::FixedValue)
RUND_COMPUTE_EXPR_BINARY(atan2, Atan2, detail::FixedValue)
#undef RUND_COMPUTE_EXPR_BINARY

template <class T>
  requires detail::FixedValue<T>
[[nodiscard]] Expr<T> mul_add_fixed(const Expr<T> &left, const Expr<T> &right,
                                    const Expr<T> &addend) {
  return detail::ExprAccess::make<T>(detail::ternary(
      detail::ExprOp::MulAddFixed, detail::ExprAccess::ref(left),
      detail::ExprAccess::ref(right), detail::ExprAccess::ref(addend)));
}

template <class T>
  requires detail::FixedValue<T>
[[nodiscard]] Expr<T> pow(const Expr<T> &base, const Expr<T> &exponent) {
  return exp(quantize<T, Rounding::NearestEven, Overflow::Saturate,
                      Approximation::Deterministic>(exponent * log(base)));
}

template <std::uint32_t Amount, class T>
  requires detail::StoredExprValue<T>
[[nodiscard]] Expr<T> shl(const Expr<T> &value) {
  static_assert(Amount < sizeof(T) * 8u, "shift amount exceeds value width");
  return detail::ExprAccess::make<T>(detail::shift(
      detail::ExprOp::ShiftLeft, detail::ExprAccess::ref(value), Amount));
}
template <std::uint32_t Amount, class T>
  requires detail::StoredExprValue<T>
[[nodiscard]] Expr<T> shr_logical(const Expr<T> &value) {
  static_assert(Amount < sizeof(T) * 8u, "shift amount exceeds value width");
  return detail::ExprAccess::make<T>(
      detail::shift(detail::ExprOp::ShiftRightLogical,
                    detail::ExprAccess::ref(value), Amount));
}
template <std::uint32_t Amount, class T>
  requires detail::SignedExprValue<T>
[[nodiscard]] Expr<T> shr_arithmetic(const Expr<T> &value) {
  static_assert(Amount < sizeof(T) * 8u, "shift amount exceeds value width");
  return detail::ExprAccess::make<T>(
      detail::shift(detail::ExprOp::ShiftRightArithmetic,
                    detail::ExprAccess::ref(value), Amount));
}

template <class Tag, class T> [[nodiscard]] auto field(const Expr<T> &value) {
  return ExprField<Tag, T>{detail::ExprAccess::ref(value)};
}

template <class Tag, class... Fields>
[[nodiscard]] auto field(const ExprRecord<Fields...> &value) {
  return ExprRecordField<Tag, Fields...>{value};
}

template <class... Fields>
  requires(detail::is_expr_field<std::remove_cvref_t<Fields>> && ...)
[[nodiscard]] auto record(Fields &&...fields) {
  return ExprRecord<std::remove_cvref_t<Fields>...>{
      std::forward<Fields>(fields)...};
}
} // namespace rund::compute
