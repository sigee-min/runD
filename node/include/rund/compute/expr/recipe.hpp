#pragma once

#include <rund/compute/expr/operation.hpp>

namespace rund::compute {
namespace detail {
template <ComputeValue T>
[[nodiscard]] constexpr std::uint64_t static_bits(const T value) noexcept {
  if constexpr (FixedValue<T> && sizeof(T) == sizeof(std::uint32_t)) {
    return static_cast<std::uint32_t>(value.raw());
  } else if constexpr (FixedValue<T>) {
    return static_cast<std::uint64_t>(value.raw());
  } else if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
    return static_cast<std::uint32_t>(value);
  } else {
    return static_cast<std::uint64_t>(value);
  }
}
template <class Tuple>
[[nodiscard]] const std::shared_ptr<ExprState> &static_state(Tuple &inputs) {
  return ExprAccess::ref(std::get<0>(inputs)).state;
}
template <class T, std::size_t I, class Function, class Tuple>
[[nodiscard]] ExprRef static_ref(const StaticExpr<T, StaticInput<I>> &,
                                 Function &, Tuple &inputs) {
  ExprRef value = retype_expr(ExprAccess::ref(std::get<I>(inputs)), type<T>());
  if constexpr (FixedValue<T>) {
    value = with_fixed_format(std::move(value), fixed_format<T>());
  }
  return value;
}
template <class T, std::size_t I, class Function, class Tuple>
[[nodiscard]] ExprRef static_ref(const StaticExpr<T, StaticCapture<I>> &,
                                 Function &function, Tuple &inputs) {
  return static_capture<I, T>(function, static_state(inputs));
}
template <class T, class Function, class Tuple>
[[nodiscard]] ExprRef static_ref(const StaticExpr<T, StaticConstant<T>> &value,
                                 Function &, Tuple &inputs) {
  if constexpr (FixedValue<T>) {
    return constant(static_state(inputs), type<T>(),
                    static_bits(value.node.value), fixed_format<T>());
  } else {
    return constant(static_state(inputs), type<T>(),
                    static_bits(value.node.value));
  }
}
template <class T, class Anchor, class Function, class Tuple>
  requires FixedValue<T>
[[nodiscard]] ExprRef
static_ref(const StaticExpr<T, StaticLiteralLike<T, Anchor>> &value,
           Function &function, Tuple &inputs) {
  ExprRef anchor = static_ref(value.node.anchor, function, inputs);
  return constant(anchor.state, type<T>(), static_bits(value.node.value),
                  fixed_literal_format<T>(anchor.fixed_format));
}
template <class T, class Anchor, class Function, class Tuple>
  requires(!FixedValue<T>)
[[nodiscard]] ExprRef
static_ref(const StaticExpr<T, StaticLiteralLike<T, Anchor>> &value,
           Function &, Tuple &inputs) {
  return constant(static_state(inputs), type<T>(),
                  static_bits(value.node.value));
}
template <class Target, Rounding Round, Overflow OverflowMode,
          Approximation ApproximationMode, class Value, class Function,
          class Tuple>
[[nodiscard]] ExprRef static_ref(
    const StaticExpr<Target, StaticQuantize<Target, Round, OverflowMode,
                                            ApproximationMode, Value>> &value,
    Function &function, Tuple &inputs) {
  ExprRef source = static_ref(value.node.value, function, inputs);
  return quantize_expr(
      std::move(source), type<Target>(),
      fixed_format<Target>(Round, OverflowMode, ApproximationMode));
}
template <class Target, class Value, class Function, class Tuple>
[[nodiscard]] ExprRef
static_ref(const StaticExpr<Target, StaticQuantizeLike<Target, Value>> &value,
           Function &function, Tuple &inputs) {
  ExprRef source = static_ref(value.node.value, function, inputs);
  const FixedFormat format = fixed_storage_format<Target>(source.fixed_format);
  return quantize_expr(std::move(source), type<Target>(), format);
}
template <class T, ExprOp Op, class Value, class Function, class Tuple>
[[nodiscard]] ExprRef
static_ref(const StaticExpr<T, StaticUnary<Op, Value>> &value,
           Function &function, Tuple &inputs) {
  ExprRef input = static_ref(value.node.value, function, inputs);
  if constexpr (Op == ExprOp::Mask) {
    return make_mask(std::move(input), type<T>());
  } else {
    input = retype_expr(std::move(input), type<T>());
    return unary(Op, std::move(input));
  }
}
template <class T, ExprOp Op, std::uint32_t Amount, class Value, class Function,
          class Tuple>
[[nodiscard]] ExprRef
static_ref(const StaticExpr<T, StaticShift<Op, Amount, Value>> &value,
           Function &function, Tuple &inputs) {
  ExprRef input = static_ref(value.node.value, function, inputs);
  input = retype_expr(std::move(input), type<T>());
  return shift(Op, std::move(input), Amount);
}
template <class T, ExprOp Op, class Left, class Right, class Function,
          class Tuple>
[[nodiscard]] ExprRef
static_ref(const StaticExpr<T, StaticBinary<Op, Left, Right>> &value,
           Function &function, Tuple &inputs) {
  ExprRef left = static_ref(value.node.left, function, inputs);
  ExprRef right = static_ref(value.node.right, function, inputs);
  left = retype_expr(std::move(left), type<T>());
  right = retype_expr(std::move(right), type<T>());
  return binary(Op, std::move(left), std::move(right));
}
template <class T, ExprOp Op, class First, class Second, class Third,
          class Function, class Tuple>
[[nodiscard]] ExprRef
static_ref(const StaticExpr<T, StaticTernary<Op, First, Second, Third>> &value,
           Function &function, Tuple &inputs) {
  ExprRef first = static_ref(value.node.first, function, inputs);
  ExprRef second = static_ref(value.node.second, function, inputs);
  ExprRef third = static_ref(value.node.third, function, inputs);
  if constexpr (Op != ExprOp::Select) {
    first = retype_expr(std::move(first), type<T>());
  }
  second = retype_expr(std::move(second), type<T>());
  third = retype_expr(std::move(third), type<T>());
  return ternary(Op, std::move(first), std::move(second), std::move(third));
}
template <class T, class Node, class Function, class Tuple>
[[nodiscard]] ExprRef static_ref(const StaticPredicate<T, Node> &value,
                                 Function &function, Tuple &inputs) {
  return static_ref(StaticExpr<T, Node>{value.node}, function, inputs);
}
template <class T, class Node, class Function, class Tuple>
[[nodiscard]] Expr<T> materialize_static(const StaticExpr<T, Node> &value,
                                         Function &function, Tuple &inputs) {
  return ExprAccess::make<T>(static_ref(value, function, inputs));
}
template <class T, class Node, class Function, class Tuple>
[[nodiscard]] Predicate<T>
materialize_static(const StaticPredicate<T, Node> &value, Function &function,
                   Tuple &inputs) {
  return ExprAccess::predicate<T>(static_ref(value, function, inputs));
}
template <class Tag, class Value, class Function, class Tuple>
[[nodiscard]] auto materialize_static(const StaticField<Tag, Value> &value,
                                      Function &function, Tuple &inputs) {
  return field<Tag>(materialize_static(value.value, function, inputs));
}
template <std::size_t I, class Field, class... Fields>
[[nodiscard]] constexpr const Field &
static_record_field(const StaticRecord<Fields...> &value) noexcept {
  return static_cast<const StaticRecordSlot<I, Field> &>(value).value;
}
template <class... Fields, class Function, class Tuple, std::size_t... I>
[[nodiscard]] auto materialize_static_record(
    const StaticRecord<Fields...> &value, Function &function, Tuple &inputs,
    std::index_sequence<I...>) {
  auto materialized = std::tuple{
      materialize_static(static_record_field<I, Fields>(value), function,
                         inputs)...};
  return std::apply(
      [](auto &&...field_value) {
        return record(
            std::forward<decltype(field_value)>(field_value)...);
      },
      std::move(materialized));
}
template <class... Fields, class Function, class Tuple>
[[nodiscard]] auto materialize_static(const StaticRecord<Fields...> &value,
                                      Function &function, Tuple &inputs) {
  return materialize_static_record(value, function, inputs,
                                   std::index_sequence_for<Fields...>{});
}
} // namespace detail
} // namespace rund::compute
