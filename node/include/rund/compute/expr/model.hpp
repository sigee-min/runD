#pragma once
#include <array>
#include <concepts>
#include <cstdint>
#include <rund/compute/abi/expression.hpp>
#include <rund/compute/fixed.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
namespace rund::compute {
template <class T> class Expr;
template <class T> class Predicate;
template <class Tag, class T> class ExprField;
template <class Tag, class... Fields> class ExprRecordField;
template <class... Fields> class ExprRecord;
template <class T>
[[nodiscard]] Expr<T> min(const Expr<T> &left, const Expr<T> &right);
template <class T, class U>
  requires std::convertible_to<U, T>
[[nodiscard]] Expr<T> min(const Expr<T> &left, U right);
template <class T>
[[nodiscard]] Expr<T> max(const Expr<T> &left, const Expr<T> &right);
template <class T, class U>
  requires std::convertible_to<U, T>
[[nodiscard]] Expr<T> max(const Expr<T> &left, U right);
template <class T>
[[nodiscard]] Expr<T> clamp(const Expr<T> &value, const Expr<T> &low,
                            const Expr<T> &high);
template <class T, class L, class H>
  requires(std::convertible_to<L, T> && std::convertible_to<H, T>)
[[nodiscard]] Expr<T> clamp(const Expr<T> &value, L low, H high);
template <class, class, class> class Flow;
template <class, class> class StageRef;
template <class, class, class> class Groups;
template <class, class> class GroupValuesRef;
template <class C, class T>
  requires(sizeof(C) == sizeof(T))
[[nodiscard]] Expr<T> select(const Predicate<C> &condition,
                             const Expr<T> &when_true,
                             const Expr<T> &when_false);
template <class C, class T, class U>
  requires(sizeof(C) == sizeof(T) && std::convertible_to<U, T>)
[[nodiscard]] Expr<T> select(const Predicate<C> &condition,
                             const Expr<T> &when_true, U when_false);
template <class C, class T, class U>
  requires(sizeof(C) == sizeof(T) && std::convertible_to<U, T>)
[[nodiscard]] Expr<T> select(const Predicate<C> &condition, U when_true,
                             const Expr<T> &when_false);
template <class T, class C>
  requires(sizeof(C) == sizeof(T))
[[nodiscard]] Expr<T> select(const Predicate<C> &condition, T when_true,
                             T when_false);
template <class T>
[[nodiscard]] Expr<std::uint32_t> mask(const Predicate<T> &condition);
template <class Target, Rounding Round = Rounding::NearestEven,
          Overflow OverflowMode = Overflow::Saturate,
          Approximation ApproximationMode = Approximation::Exact, class Source>
  requires(detail::FixedValue<Target> && detail::FixedValue<Source>)
[[nodiscard]] Expr<Target> quantize(const Expr<Source> &value);
namespace detail {
template <std::size_t I> struct StaticInput final {};
template <std::size_t I> struct StaticCapture final {};
template <class T> struct StaticConstant final {
  T value;
};
template <class T, class Anchor> struct StaticLiteralLike final {
  Anchor anchor;
  T value;
};
template <ExprOp Op, class Value> struct StaticUnary final {
  Value value;
};
template <class Target, Rounding Round, Overflow OverflowMode,
          Approximation ApproximationMode, class Value>
struct StaticQuantize final {
  Value value;
};
template <class Target, class Value> struct StaticQuantizeLike final {
  Value value;
};
template <class T>
  requires FixedValue<T>
[[nodiscard]] constexpr FixedFormat
fixed_literal_format(const FixedFormat anchor) noexcept {
  FixedFormat format = fixed_format<T>();
  if (anchor.integer_bits != 0u) {
    format.rounding = anchor.rounding;
    format.overflow = anchor.overflow;
  }
  return format;
}
template <class T>
  requires FixedValue<T>
[[nodiscard]] constexpr FixedFormat
fixed_storage_format(const FixedFormat anchor) noexcept {
  FixedFormat format = fixed_literal_format<T>(anchor);
  if (anchor.integer_bits != 0u) {
    format.approximation = anchor.approximation;
  }
  return format;
}
template <ExprOp Op, std::uint32_t Amount, class Value>
struct StaticShift final {
  Value value;
};
template <ExprOp Op, class Left, class Right> struct StaticBinary final {
  Left left;
  Right right;
};
template <ExprOp Op, class First, class Second, class Third>
struct StaticTernary final {
  First first;
  Second second;
  Third third;
};
template <class T, class Node> struct StaticPredicate final {
  Node node;
  [[nodiscard]] constexpr auto operator!() const {
    using Self = StaticPredicate<T, Node>;
    return StaticPredicate<T, StaticUnary<ExprOp::PredicateNot, Self>>{{*this}};
  }
  template <class U, class Right>
  [[nodiscard]] constexpr auto
  operator&&(const StaticPredicate<U, Right> &right) const {
    using Left = StaticPredicate<T, Node>;
    using Other = StaticPredicate<U, Right>;
    return StaticPredicate<T, StaticBinary<ExprOp::PredicateAnd, Left, Other>>{
        {*this, right}};
  }
  template <class U, class Right>
  [[nodiscard]] constexpr auto
  operator||(const StaticPredicate<U, Right> &right) const {
    using Left = StaticPredicate<T, Node>;
    using Other = StaticPredicate<U, Right>;
    return StaticPredicate<T, StaticBinary<ExprOp::PredicateOr, Left, Other>>{
        {*this, right}};
  }
};
template <class T, class Node> struct StaticExpr final {
  using Value = T;
  Node node;

  [[nodiscard]] constexpr auto operator-() const {
    using Self = StaticExpr<T, Node>;
    return StaticExpr<T, StaticUnary<ExprOp::Negate, Self>>{{*this}};
  }
#define RUND_COMPUTE_STATIC_BINARY(symbol, operation)                          \
  template <class U, class Right>                                              \
    requires((std::integral<T> && std::integral<U>) || std::same_as<T, U>)     \
  [[nodiscard]] constexpr auto operator symbol(                                \
      const StaticExpr<U, Right> &right) const {                               \
    using Result = std::conditional_t<std::integral<T> && std::integral<U>,    \
                                      std::common_type_t<T, U>, T>;            \
    using LeftValue = StaticExpr<T, Node>;                                     \
    using RightValue = StaticExpr<U, Right>;                                   \
    return StaticExpr<Result,                                                  \
                      StaticBinary<ExprOp::operation, LeftValue, RightValue>>{ \
        {*this, right}};                                                       \
  }                                                                            \
  template <class U>                                                           \
    requires std::convertible_to<U, T>                                         \
  [[nodiscard]] constexpr auto operator symbol(const U right) const {          \
    using LeftValue = StaticExpr<T, Node>;                                     \
    using RightValue = StaticExpr<T, StaticLiteralLike<T, LeftValue>>;         \
    return StaticExpr<T,                                                       \
                      StaticBinary<ExprOp::operation, LeftValue, RightValue>>{ \
        {*this, RightValue{StaticLiteralLike<T, LeftValue>{                    \
                    *this, static_cast<T>(right)}}}};                          \
  }
  RUND_COMPUTE_STATIC_BINARY(+, Add)
  RUND_COMPUTE_STATIC_BINARY(-, Subtract)
  RUND_COMPUTE_STATIC_BINARY(*, Multiply)
  RUND_COMPUTE_STATIC_BINARY(/, Divide)
  RUND_COMPUTE_STATIC_BINARY(&, BitAnd)
  RUND_COMPUTE_STATIC_BINARY(|, BitOr)
  RUND_COMPUTE_STATIC_BINARY(^, BitXor)
#undef RUND_COMPUTE_STATIC_BINARY
#define RUND_COMPUTE_STATIC_COMPARE(symbol, operation)                         \
  template <class U, class Right>                                              \
    requires((std::integral<T> && std::integral<U>) || std::same_as<T, U>)     \
  [[nodiscard]] constexpr auto operator symbol(                                \
      const StaticExpr<U, Right> &right) const {                               \
    using Result = std::conditional_t<std::integral<T> && std::integral<U>,    \
                                      std::common_type_t<T, U>, T>;            \
    using LeftValue = StaticExpr<T, Node>;                                     \
    using RightValue = StaticExpr<U, Right>;                                   \
    return StaticPredicate<                                                    \
        Result, StaticBinary<ExprOp::operation, LeftValue, RightValue>>{       \
        {*this, right}};                                                       \
  }                                                                            \
  template <class U>                                                           \
    requires std::convertible_to<U, T>                                         \
  [[nodiscard]] constexpr auto operator symbol(const U right) const {          \
    using LeftValue = StaticExpr<T, Node>;                                     \
    using RightValue = StaticExpr<T, StaticLiteralLike<T, LeftValue>>;         \
    return StaticPredicate<                                                    \
        T, StaticBinary<ExprOp::operation, LeftValue, RightValue>>{            \
        {*this, RightValue{StaticLiteralLike<T, LeftValue>{                    \
                    *this, static_cast<T>(right)}}}};                          \
  }
  RUND_COMPUTE_STATIC_COMPARE(==, Equal)
  RUND_COMPUTE_STATIC_COMPARE(!=, NotEqual)
  RUND_COMPUTE_STATIC_COMPARE(<, Less)
  RUND_COMPUTE_STATIC_COMPARE(<=, LessEqual)
  RUND_COMPUTE_STATIC_COMPARE(>, Greater)
  RUND_COMPUTE_STATIC_COMPARE(>=, GreaterEqual)
#undef RUND_COMPUTE_STATIC_COMPARE
};
template <class T> inline constexpr bool is_static_expr = false;
template <class T, class Node>
inline constexpr bool is_static_expr<StaticExpr<T, Node>> = true;
template <class Tag, class Value> struct StaticField final {
  Value value;
};
template <std::size_t I, class Field> struct StaticRecordSlot {
  Field value;
};
template <class Index, class... Fields> struct StaticRecordStorage;
template <std::size_t... I, class... Fields>
struct StaticRecordStorage<std::index_sequence<I...>, Fields...>
    : StaticRecordSlot<I, Fields>... {
  constexpr explicit StaticRecordStorage(Fields... fields)
      : StaticRecordSlot<I, Fields>{std::move(fields)}... {}
};
template <class... Fields>
struct StaticRecord final
    : StaticRecordStorage<std::index_sequence_for<Fields...>, Fields...> {
  using Storage =
      StaticRecordStorage<std::index_sequence_for<Fields...>, Fields...>;
  constexpr explicit StaticRecord(Fields... fields)
      : Storage(std::move(fields)...) {}
};
template <class T> inline constexpr bool is_static_field = false;
template <class Tag, class Value>
inline constexpr bool is_static_field<StaticField<Tag, Value>> = true;
template <class T, std::size_t I> struct StaticArg;
template <class T, std::size_t I> struct StaticArg<Expr<T>, I> final {
  using Type = StaticExpr<T, StaticInput<I>>;
};
template <class T, std::size_t I>
using StaticArgT = typename StaticArg<std::remove_cvref_t<T>, I>::Type;
template <std::size_t I, class T, class Function>
[[nodiscard]] ExprRef static_capture(Function &function,
                                     const std::shared_ptr<ExprState> &state);

struct ExprAccess;
struct ExprRecordAccess;
template <class> struct ExprValue;
template <class T> struct ExprValue<Expr<T>> final {
  using type = T;
};
template <class T>
using ExprValueT = typename ExprValue<std::remove_cvref_t<T>>::type;
template <class T>
concept ComputeExpr =
    requires { typename ExprValue<std::remove_cvref_t<T>>::type; };
template <class T> inline constexpr bool is_expr_record = false;
template <class... Fields>
inline constexpr bool is_expr_record<ExprRecord<Fields...>> = true;
template <class T> inline constexpr bool is_expr_field = false;
template <class Tag, class T>
inline constexpr bool is_expr_field<ExprField<Tag, T>> = true;
template <class Tag, class... Fields>
inline constexpr bool is_expr_field<ExprRecordField<Tag, Fields...>> = true;
} // namespace detail

} // namespace rund::compute
