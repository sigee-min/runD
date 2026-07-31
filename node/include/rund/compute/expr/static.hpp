#pragma once

#include <rund/compute/expr/model.hpp>

namespace rund::compute {
template <class C, class Condition, class T, class True, class False>
[[nodiscard]] constexpr auto
select(const detail::StaticPredicate<C, Condition> &condition,
       const detail::StaticExpr<T, True> &when_true,
       const detail::StaticExpr<T, False> &when_false) {
  return detail::StaticExpr<
      T, detail::StaticTernary<
             detail::ExprOp::Select, detail::StaticPredicate<C, Condition>,
             detail::StaticExpr<T, True>, detail::StaticExpr<T, False>>>{
      {condition, when_true, when_false}};
}
template <class C, class Condition, class T, class True, class U>
  requires std::convertible_to<U, T>
[[nodiscard]] constexpr auto
select(const detail::StaticPredicate<C, Condition> &condition,
       const detail::StaticExpr<T, True> &when_true, const U when_false) {
  using Anchor = detail::StaticExpr<T, True>;
  using False = detail::StaticExpr<T, detail::StaticLiteralLike<T, Anchor>>;
  return detail::StaticExpr<
      T, detail::StaticTernary<detail::ExprOp::Select,
                               detail::StaticPredicate<C, Condition>,
                               detail::StaticExpr<T, True>, False>>{
      {condition, when_true,
       False{detail::StaticLiteralLike<T, Anchor>{
           when_true, static_cast<T>(when_false)}}}};
}
template <class C, class Condition, class T, class False, class U>
  requires std::convertible_to<U, T>
[[nodiscard]] constexpr auto
select(const detail::StaticPredicate<C, Condition> &condition,
       const U when_true, const detail::StaticExpr<T, False> &when_false) {
  using Anchor = detail::StaticExpr<T, False>;
  using True = detail::StaticExpr<T, detail::StaticLiteralLike<T, Anchor>>;
  return detail::StaticExpr<
      T, detail::StaticTernary<detail::ExprOp::Select,
                               detail::StaticPredicate<C, Condition>, True,
                               detail::StaticExpr<T, False>>>{
      {condition,
       True{detail::StaticLiteralLike<T, Anchor>{when_false,
                                                 static_cast<T>(when_true)}},
       when_false}};
}
template <class T, class C, class Condition>
  requires(!detail::is_static_expr<std::remove_cvref_t<T>>)
[[nodiscard]] constexpr auto
select(const detail::StaticPredicate<C, Condition> &condition,
       const T when_true, const T when_false) {
  using Constant = detail::StaticExpr<T, detail::StaticConstant<T>>;
  return detail::StaticExpr<
      T, detail::StaticTernary<detail::ExprOp::Select,
                               detail::StaticPredicate<C, Condition>, Constant,
                               Constant>>{
      {condition, {{when_true}}, {{when_false}}}};
}
template <class T, class Condition>
[[nodiscard]] constexpr auto
mask(const detail::StaticPredicate<T, Condition> &condition) {
  return detail::StaticExpr<
      std::uint32_t,
      detail::StaticUnary<detail::ExprOp::Mask,
                          detail::StaticPredicate<T, Condition>>>{{condition}};
}
template <class Target, Rounding Round = Rounding::NearestEven,
          Overflow OverflowMode = Overflow::Saturate,
          Approximation ApproximationMode = Approximation::Exact, class Source,
          class Node>
  requires(detail::FixedValue<Target> && detail::FixedValue<Source>)
[[nodiscard]] constexpr auto
quantize(const detail::StaticExpr<Source, Node> &value) {
  using Input = detail::StaticExpr<Source, Node>;
  return detail::StaticExpr<Target,
                            detail::StaticQuantize<Target, Round, OverflowMode,
                                                   ApproximationMode, Input>>{
      {value}};
}
template <class T, class Left, class U, class Right>
  requires((std::integral<T> && std::integral<U>) || std::same_as<T, U>)
[[nodiscard]] constexpr auto min(const detail::StaticExpr<T, Left> &left,
                                 const detail::StaticExpr<U, Right> &right) {
  using Result = std::conditional_t<std::integral<T> && std::integral<U>,
                                    std::common_type_t<T, U>, T>;
  return detail::StaticExpr<
      Result,
      detail::StaticBinary<detail::ExprOp::Min, detail::StaticExpr<T, Left>,
                           detail::StaticExpr<U, Right>>>{{left, right}};
}
template <class T, class Left, class U>
  requires(std::convertible_to<U, T> &&
           !detail::is_static_expr<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto min(const detail::StaticExpr<T, Left> &left,
                                 const U right) {
  using Anchor = detail::StaticExpr<T, Left>;
  using Literal = detail::StaticExpr<T, detail::StaticLiteralLike<T, Anchor>>;
  return detail::StaticExpr<
      T, detail::StaticBinary<detail::ExprOp::Min, detail::StaticExpr<T, Left>,
                              Literal>>{
      {left, Literal{detail::StaticLiteralLike<T, Anchor>{
                 left, static_cast<T>(right)}}}};
}
template <class T, class Left, class U, class Right>
  requires((std::integral<T> && std::integral<U>) || std::same_as<T, U>)
[[nodiscard]] constexpr auto max(const detail::StaticExpr<T, Left> &left,
                                 const detail::StaticExpr<U, Right> &right) {
  using Result = std::conditional_t<std::integral<T> && std::integral<U>,
                                    std::common_type_t<T, U>, T>;
  return detail::StaticExpr<
      Result,
      detail::StaticBinary<detail::ExprOp::Max, detail::StaticExpr<T, Left>,
                           detail::StaticExpr<U, Right>>>{{left, right}};
}
template <class T, class Left, class U>
  requires(std::convertible_to<U, T> &&
           !detail::is_static_expr<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto max(const detail::StaticExpr<T, Left> &left,
                                 const U right) {
  using Anchor = detail::StaticExpr<T, Left>;
  using Literal = detail::StaticExpr<T, detail::StaticLiteralLike<T, Anchor>>;
  return detail::StaticExpr<
      T, detail::StaticBinary<detail::ExprOp::Max, detail::StaticExpr<T, Left>,
                              Literal>>{
      {left, Literal{detail::StaticLiteralLike<T, Anchor>{
                 left, static_cast<T>(right)}}}};
}
template <class T, class Node, class Low, class High>
[[nodiscard]] constexpr auto clamp(const detail::StaticExpr<T, Node> &value,
                                   const detail::StaticExpr<T, Low> &low,
                                   const detail::StaticExpr<T, High> &high) {
  return detail::StaticExpr<
      T, detail::StaticTernary<
             detail::ExprOp::Clamp, detail::StaticExpr<T, Node>,
             detail::StaticExpr<T, Low>, detail::StaticExpr<T, High>>>{
      {value, low, high}};
}
template <class T, class Node, class Low, class H>
  requires std::convertible_to<H, T>
[[nodiscard]] constexpr auto clamp(const detail::StaticExpr<T, Node> &value,
                                   const detail::StaticExpr<T, Low> &low,
                                   const H high) {
  using Anchor = detail::StaticExpr<T, Node>;
  using Literal = detail::StaticExpr<T, detail::StaticLiteralLike<T, Anchor>>;
  return clamp(value, low,
               Literal{detail::StaticLiteralLike<T, Anchor>{
                   value, static_cast<T>(high)}});
}
template <class T, class Node, class L, class High>
  requires std::convertible_to<L, T>
[[nodiscard]] constexpr auto clamp(const detail::StaticExpr<T, Node> &value,
                                   const L low,
                                   const detail::StaticExpr<T, High> &high) {
  using Anchor = detail::StaticExpr<T, Node>;
  using Literal = detail::StaticExpr<T, detail::StaticLiteralLike<T, Anchor>>;
  return clamp(
      value,
      Literal{detail::StaticLiteralLike<T, Anchor>{value, static_cast<T>(low)}},
      high);
}
template <class T, class Node, class L, class H>
  requires(std::convertible_to<L, T> && std::convertible_to<H, T>)
[[nodiscard]] constexpr auto clamp(const detail::StaticExpr<T, Node> &value,
                                   const L low, const H high) {
  using Anchor = detail::StaticExpr<T, Node>;
  using Literal = detail::StaticExpr<T, detail::StaticLiteralLike<T, Anchor>>;
  return clamp(
      value,
      Literal{detail::StaticLiteralLike<T, Anchor>{value, static_cast<T>(low)}},
      Literal{
          detail::StaticLiteralLike<T, Anchor>{value, static_cast<T>(high)}});
}
template <class Tag, class T, class Node>
[[nodiscard]] constexpr auto field(const detail::StaticExpr<T, Node> &value) {
  return detail::StaticField<Tag, detail::StaticExpr<T, Node>>{value};
}
template <class Tag, class... Fields>
[[nodiscard]] constexpr auto
field(const detail::StaticRecord<Fields...> &value) {
  return detail::StaticField<Tag, detail::StaticRecord<Fields...>>{value};
}
template <class... Fields>
  requires(detail::is_static_field<std::remove_cvref_t<Fields>> && ...)
[[nodiscard]] constexpr auto record(Fields &&...fields) {
  return detail::StaticRecord<std::remove_cvref_t<Fields>...>{
      std::forward<Fields>(fields)...};
}

} // namespace rund::compute
