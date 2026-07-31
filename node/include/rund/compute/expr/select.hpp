#pragma once

#include <rund/compute/expr/operation.hpp>

namespace rund::compute {
template <class T> Expr<T> min(const Expr<T> &left, const Expr<T> &right) {
  return Expr<T>{detail::binary(detail::ExprOp::Min, left.ref_, right.ref_)};
}
template <class T, class U>
  requires std::convertible_to<U, T>
Expr<T> min(const Expr<T> &left, const U right) {
  return min(left, left.literal(static_cast<T>(right)));
}
template <class T> Expr<T> max(const Expr<T> &left, const Expr<T> &right) {
  return Expr<T>{detail::binary(detail::ExprOp::Max, left.ref_, right.ref_)};
}
template <class T, class U>
  requires std::convertible_to<U, T>
Expr<T> max(const Expr<T> &left, const U right) {
  return max(left, left.literal(static_cast<T>(right)));
}
template <class T>
Expr<T> clamp(const Expr<T> &value, const Expr<T> &low, const Expr<T> &high) {
  return Expr<T>{
      detail::ternary(detail::ExprOp::Clamp, value.ref_, low.ref_, high.ref_)};
}
template <class T, class L, class H>
  requires(std::convertible_to<L, T> && std::convertible_to<H, T>)
Expr<T> clamp(const Expr<T> &value, const L low, const H high) {
  return clamp(value, value.literal(static_cast<T>(low)),
               value.literal(static_cast<T>(high)));
}
template <class C, class T>
  requires(sizeof(C) == sizeof(T))
Expr<T> select(const Predicate<C> &condition, const Expr<T> &when_true,
               const Expr<T> &when_false) {
  return Expr<T>{detail::ternary(detail::ExprOp::Select, condition.ref_,
                                 when_true.ref_, when_false.ref_)};
}
template <class C, class T, class U>
  requires(sizeof(C) == sizeof(T) && std::convertible_to<U, T>)
Expr<T> select(const Predicate<C> &condition, const Expr<T> &when_true,
               const U when_false) {
  return select(condition, when_true,
                when_true.literal(static_cast<T>(when_false)));
}
template <class C, class T, class U>
  requires(sizeof(C) == sizeof(T) && std::convertible_to<U, T>)
Expr<T> select(const Predicate<C> &condition, const U when_true,
               const Expr<T> &when_false) {
  return select(condition, when_false.literal(static_cast<T>(when_true)),
                when_false);
}
template <class T, class C>
  requires(sizeof(C) == sizeof(T))
Expr<T> select(const Predicate<C> &condition, const T when_true,
               const T when_false) {
  std::uint64_t true_bits = 0u;
  std::uint64_t false_bits = 0u;
  if constexpr (detail::FixedValue<T> && sizeof(T) == sizeof(std::uint32_t)) {
    true_bits = static_cast<std::uint32_t>(when_true.raw());
    false_bits = static_cast<std::uint32_t>(when_false.raw());
  } else if constexpr (detail::FixedValue<T>) {
    true_bits = static_cast<std::uint64_t>(when_true.raw());
    false_bits = static_cast<std::uint64_t>(when_false.raw());
  } else if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
    true_bits = static_cast<std::uint32_t>(when_true);
    false_bits = static_cast<std::uint32_t>(when_false);
  } else {
    true_bits = static_cast<std::uint64_t>(when_true);
    false_bits = static_cast<std::uint64_t>(when_false);
  }
  detail::FixedFormat format{};
  if constexpr (detail::FixedValue<T>) {
    format = detail::fixed_format<T>();
  }
  return Expr<T>{
      detail::ternary(detail::ExprOp::Select, condition.ref_,
                      detail::constant(condition.ref_.state, detail::type<T>(),
                                       true_bits, format),
                      detail::constant(condition.ref_.state, detail::type<T>(),
                                       false_bits, format))};
}
template <class T> Expr<std::uint32_t> mask(const Predicate<T> &condition) {
  return Expr<std::uint32_t>{detail::make_mask(condition.ref_)};
}
} // namespace rund::compute
