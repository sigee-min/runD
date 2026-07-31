#pragma once

#include <rund/compute/flow/stage.hpp>

namespace rund::compute {
template <class T, class Card>
template <class RightCard, class LeftKeyFn, class RightKeyFn, class EmitFn>
[[nodiscard]] auto StageRef<T, Card>::join(const MaxMatches bound,
                                           const StageRef<T, RightCard> &right,
                                           LeftKeyFn &&left_key_function,
                                           RightKeyFn &&right_key_function,
                                           EmitFn &&emit_function) const
  requires((std::same_as<Card, stage::Exact> ||
            detail::is_bounded_stage<Card>) &&
           (std::same_as<RightCard, stage::Exact> ||
            detail::is_bounded_stage<RightCard>) &&
           detail::IntegerValue<T>)
{
  using Count = CountFor<T>;
  using LeftCount = detail::ResidentCountT<T, Card>;
  using RightCount = detail::ResidentCountT<T, RightCard>;
  auto emit_expressions = detail::make_expr();
  Expr<T> emit_left{
      detail::flow_expression_input<T>(state_, emit_expressions, value_, 0u)};
  Expr<T> emit_right{detail::flow_expression_input<T>(state_, emit_expressions,
                                                      right.value_, 1u)};
  auto emit = detail::element(emit_function, emit_left, emit_right);
  using Emit = std::remove_cvref_t<decltype(emit)>;
  static_assert(detail::ComputeExpr<Emit> || detail::is_expr_record<Emit>,
                "compute join emit must return an expression or record");
  if constexpr (detail::ComputeExpr<Emit>) {
    static_assert(sizeof(detail::ExprValueT<Emit>) == sizeof(T),
                  "compute join emit must preserve scalar width");
  } else {
    static_assert(
        detail::ExprRecordStage<Emit,
                                stage::Bounded<Count>>::template accepts<T>,
        "all join record fields must preserve scalar width");
  }

  const std::size_t left_count = detail::flow_value_count(state_, value_);
  const std::size_t right_count =
      detail::flow_value_count(state_, right.value_);
  if (state_ != right.state_ || bound.value == 0u ||
      left_count > std::numeric_limits<std::uint32_t>::max() ||
      (right_count != 0u &&
       left_count > std::numeric_limits<std::size_t>::max() / right_count) ||
      left_count * right_count > std::numeric_limits<std::uint32_t>::max() ||
      left_count > std::numeric_limits<std::size_t>::max() / bound.value ||
      left_count * bound.value > std::numeric_limits<std::uint32_t>::max()) {
    detail::flow_reject(state_, Reason::JoinCapacity);
    return detail::bounded_emit_reject<Emit, Count>(state_);
  }

  const auto constant_count = [&](auto count_type, const std::size_t value,
                                  const std::string_view name) {
    using C = typename decltype(count_type)::type;
    const std::uint32_t input =
        detail::flow_index(state_, detail::type<C>(), 1u);
    auto expressions = detail::make_expr();
    Expr<C> zero{
        detail::flow_expression_input<C>(state_, expressions, input, 0u)};
    const Expr<C> logical = zero + static_cast<C>(value);
    const std::array inputs{input};
    return detail::flow_map_value(state_, inputs, name, logical.ref_);
  };
  const std::uint32_t left_logical = [&] {
    if constexpr (detail::is_bounded_stage<Card>) {
      return count_;
    }
    return constant_count(std::type_identity<LeftCount>{}, left_count,
                          "join-left-count");
  }();
  const std::uint32_t right_logical = [&] {
    if constexpr (detail::is_bounded_stage<RightCard>) {
      return right.count_;
    }
    return constant_count(std::type_identity<RightCount>{}, right_count,
                          "join-right-count");
  }();

  if (left_count == 0u) {
    const std::uint32_t output =
        detail::flow_retype(state_, value_, detail::type<T>());
    const std::uint32_t scalar_zero =
        detail::flow_index(state_, detail::type<Count>(), 1u);
    const std::uint32_t count = detail::flow_unary_value(
        state_, scalar_zero, detail::Primitive::Reduce, detail::type<Count>(),
        1u, {.mode = static_cast<std::uint32_t>(Reduce::Sum)});
    const std::array empty_inputs{output, output};
    return detail::bounded_emit<Emit, Count>(state_, empty_inputs, "join-emit",
                                             emit, count);
  }

  if (right_count == 0u) {
    const std::size_t capacity = left_count * bound.value;
    const std::uint32_t output =
        detail::flow_index(state_, detail::type<T>(), capacity);
    const std::uint32_t scalar_zero =
        detail::flow_index(state_, detail::type<Count>(), 1u);
    const std::uint32_t count = detail::flow_unary_value(
        state_, scalar_zero, detail::Primitive::Reduce, detail::type<Count>(),
        1u, {.mode = static_cast<std::uint32_t>(Reduce::Sum)});
    const std::array empty_inputs{output, output};
    return detail::bounded_emit<Emit, Count>(state_, empty_inputs, "join-emit",
                                             emit, count);
  }

  auto left_key_expressions = detail::make_expr();
  Expr<T> left_value{detail::flow_expression_input<T>(
      state_, left_key_expressions, value_, 0u)};
  auto left_key = detail::element(left_key_function, left_value);
  auto right_key_expressions = detail::make_expr();
  Expr<T> right_value{detail::flow_expression_input<T>(
      state_, right_key_expressions, right.value_, 0u)};
  auto right_key = detail::element(right_key_function, right_value);
  static_assert(detail::ComputeExpr<decltype(left_key)> &&
                    detail::ComputeExpr<decltype(right_key)>,
                "compute join keys must be compute expressions");
  using Key = detail::ExprValueT<decltype(left_key)>;
  static_assert(std::same_as<Key, detail::ExprValueT<decltype(right_key)>> &&
                    detail::IntegerValue<Key> && sizeof(Key) == sizeof(T),
                "compute join keys must have the same scalar width");
  const std::array left_key_inputs{value_};
  const std::uint32_t left_keys = detail::flow_map_value(
      state_, left_key_inputs, "join-left-key", left_key.ref_);
  const std::array right_key_inputs{right.value_};
  const std::uint32_t right_keys = detail::flow_map_value(
      state_, right_key_inputs, "join-right-key", right_key.ref_);

  const std::size_t cross_count = left_count * right_count;
  const StageRef<std::uint32_t, stage::Exact> slots{
      state_, detail::flow_index(state_, detail::Type::U32, cross_count)};
  const auto left_indices = slots.map(
      "join-left-index", capture([](auto i, auto width) { return i / width; },
                                 static_cast<std::uint32_t>(right_count)));
  const auto right_indices = slots.map(
      "join-right-index",
      capture([](auto i, auto width) { return i - (i / width) * width; },
              static_cast<std::uint32_t>(right_count)));
  const std::array left_value_inputs{value_, left_indices.value_};
  const std::uint32_t left_values = detail::flow_binary_values(
      state_, detail::Primitive::Gather, left_value_inputs, detail::type<T>(),
      cross_count, {});
  const std::array right_value_inputs{right.value_, right_indices.value_};
  const std::uint32_t right_values = detail::flow_binary_values(
      state_, detail::Primitive::Gather, right_value_inputs, detail::type<T>(),
      cross_count, {});
  const std::array left_match_inputs{left_keys, left_indices.value_};
  const std::uint32_t matched_left_keys = detail::flow_binary_values(
      state_, detail::Primitive::Gather, left_match_inputs, detail::type<Key>(),
      cross_count, {});
  const std::array right_match_inputs{right_keys, right_indices.value_};
  const std::uint32_t matched_right_keys = detail::flow_binary_values(
      state_, detail::Primitive::Gather, right_match_inputs,
      detail::type<Key>(), cross_count, {});

  auto match_expressions = detail::make_expr();
  Expr<Key> first_key{detail::flow_expression_input<Key>(
      state_, match_expressions, matched_left_keys, 0u)};
  Expr<Key> second_key{detail::flow_expression_input<Key>(
      state_, match_expressions, matched_right_keys, 1u)};
  const Expr<Key> typed_match = select(first_key == second_key, Key{1}, Key{0});
  const std::array match_inputs{matched_left_keys, matched_right_keys};
  const StageRef<Key, stage::Exact> key_matches{
      state_, detail::flow_map_value(state_, match_inputs, "join-match",
                                     typed_match.ref_)};
  const StageRef<LeftCount, stage::Exact> typed_left_slots{
      state_,
      detail::flow_index(state_, detail::type<LeftCount>(), cross_count)};
  const auto typed_left_indices = typed_left_slots.map(
      "join-typed-left-index",
      capture([](auto index, auto width) { return index / width; },
              static_cast<LeftCount>(right_count)));
  const StageRef<RightCount, stage::Exact> typed_right_slots{
      state_,
      detail::flow_index(state_, detail::type<RightCount>(), cross_count)};
  const auto typed_right_indices = typed_right_slots.map(
      "join-typed-right-index",
      capture([](auto index,
                 auto width) { return index - (index / width) * width; },
              static_cast<RightCount>(right_count)));
  const StageRef<LeftCount, stage::Scalar> left_size{state_, left_logical};
  const StageRef<RightCount, stage::Scalar> right_size{state_, right_logical};
  const auto left_active = typed_left_indices.combine(
      "join-left-active", left_size, [](auto index, auto logical) {
        return detail::count_mask<Count>(index < logical);
      });
  const auto right_active = typed_right_indices.combine(
      "join-right-active", right_size, [](auto index, auto logical) {
        return detail::count_mask<Count>(index < logical);
      });
  const auto active =
      left_active.combine("join-active", right_active,
                          [](auto left, auto right) { return left & right; });
  const StageRef<Key, stage::Exact> typed_active{
      state_, detail::flow_retype(state_, active.value_, detail::type<Key>())};
  const auto active_matches = key_matches.combine(
      "join-active-match", typed_active,
      [](auto matched, auto selected) { return matched & selected; });
  const std::uint32_t matches =
      detail::flow_retype(state_, active_matches.value_, detail::type<Count>());

  const std::uint32_t match_order = detail::flow_unary_value(
      state_, matched_left_keys, detail::Primitive::Argsort, detail::Type::U32,
      cross_count, {.flag = true});
  const std::array ordered_match_inputs{matches, match_order};
  const std::uint32_t ordered_matches = detail::flow_binary_values(
      state_, detail::Primitive::Gather, ordered_match_inputs,
      detail::type<Count>(), cross_count, {});
  const std::array ordered_left_inputs{left_values, match_order};
  const std::uint32_t ordered_left = detail::flow_binary_values(
      state_, detail::Primitive::Gather, ordered_left_inputs, detail::type<T>(),
      cross_count, {});
  const std::array ordered_right_inputs{right_values, match_order};
  const std::uint32_t ordered_right = detail::flow_binary_values(
      state_, detail::Primitive::Gather, ordered_right_inputs,
      detail::type<T>(), cross_count, {});

  const auto heads =
      slots.map("join-head", capture(
                                 [](auto index, auto width) {
                                   const auto local =
                                       index - (index / width) * width;
                                   return select(local == 0u, 1u, 0u);
                                 },
                                 static_cast<std::uint32_t>(right_count)));
  const std::array segment_inputs{matches, heads.value_};
  const std::uint32_t segment_counts = detail::flow_binary_values(
      state_, detail::Primitive::SegmentedReduce, segment_inputs,
      detail::type<Count>(), cross_count,
      {.mode = static_cast<std::uint32_t>(Reduce::Sum)});
  const std::uint32_t group_indices =
      detail::flow_index(state_, detail::Type::U32, left_count);
  const std::array group_count_inputs{segment_counts, group_indices};
  const std::uint32_t left_match_counts = detail::flow_binary_values(
      state_, detail::Primitive::Gather, group_count_inputs,
      detail::type<Count>(), left_count, {});
  const std::uint32_t max_matches = detail::flow_unary_value(
      state_, left_match_counts, detail::Primitive::Reduce,
      detail::type<Count>(), 1u,
      {.mode = static_cast<std::uint32_t>(Reduce::Max)});
  auto invalid_expressions = detail::make_expr();
  Expr<Count> maximum{detail::flow_expression_input<Count>(
      state_, invalid_expressions, max_matches, 0u)};
  const Expr<Count> invalid =
      select(maximum > static_cast<Count>(bound.value), Count{1}, Count{0});
  const std::array invalid_inputs{max_matches};
  const std::uint32_t invalid_value = detail::flow_map_value(
      state_, invalid_inputs, "join-invalid", invalid.ref_);
  auto validation_expressions = detail::make_expr();
  Expr<Count> invalid_count{detail::flow_expression_input<Count>(
      state_, validation_expressions, invalid_value, 0u)};
  const Expr<Count> validation_count = invalid_count + Count{1};
  const std::array validation_count_inputs{invalid_value};
  const std::uint32_t checked_size =
      detail::flow_map_value(state_, validation_count_inputs,
                             "join-validation-count", validation_count.ref_);
  const std::uint32_t validation_input =
      detail::flow_index(state_, detail::type<Count>(), 1u);
  const std::array validation_inputs{validation_input, checked_size};
  const std::uint32_t validation = detail::flow_binary_values(
      state_, detail::Primitive::Reduce, validation_inputs,
      detail::type<Count>(), 1u,
      {.mode = static_cast<std::uint32_t>(Reduce::Sum)});

  const std::uint32_t total = detail::flow_unary_value(
      state_, matches, detail::Primitive::Reduce, detail::type<Count>(), 1u,
      {.mode = static_cast<std::uint32_t>(Reduce::Sum)});
  auto checked_expressions = detail::make_expr();
  Expr<Count> total_value{detail::flow_expression_input<Count>(
      state_, checked_expressions, total, 0u)};
  Expr<Count> validation_value{detail::flow_expression_input<Count>(
      state_, checked_expressions, validation, 1u)};
  const Expr<Count> checked = total_value + validation_value;
  const std::array checked_inputs{total, validation};
  const std::uint32_t checked_count = detail::flow_map_value(
      state_, checked_inputs, "join-count", checked.ref_);

  auto reject_expressions = detail::make_expr();
  Expr<Count> matched{detail::flow_expression_input<Count>(
      state_, reject_expressions, ordered_matches, 0u)};
  const Expr<Count> rejected = select(matched != Count{0}, Count{0}, Count{1});
  const std::array reject_inputs{ordered_matches};
  const std::uint32_t reject_mask = detail::flow_map_value(
      state_, reject_inputs, "join-rejected", rejected.ref_);
  const std::array packed_left_inputs{reject_mask, ordered_left};
  const std::uint32_t packed_left = detail::flow_binary_values(
      state_, detail::Primitive::Partition, packed_left_inputs,
      detail::type<T>(), cross_count, {});
  const std::array packed_right_inputs{reject_mask, ordered_right};
  const std::uint32_t packed_right = detail::flow_binary_values(
      state_, detail::Primitive::Partition, packed_right_inputs,
      detail::type<T>(), cross_count, {});

  const std::size_t width =
      right_count < bound.value ? right_count : bound.value;
  const std::size_t capacity = left_count * width;
  const std::uint32_t output_indices =
      detail::flow_index(state_, detail::Type::U32, capacity);
  const std::array bounded_left_inputs{packed_left, output_indices};
  const StageRef<T, stage::Exact> bounded_left{
      state_, detail::flow_binary_values(state_, detail::Primitive::Gather,
                                         bounded_left_inputs, detail::type<T>(),
                                         capacity, {})};
  const std::array bounded_right_inputs{packed_right, output_indices};
  const StageRef<T, stage::Exact> bounded_right{
      state_, detail::flow_binary_values(state_, detail::Primitive::Gather,
                                         bounded_right_inputs,
                                         detail::type<T>(), capacity, {})};
  const std::array emit_inputs{bounded_left.value_, bounded_right.value_};
  return detail::bounded_emit<Emit, Count>(state_, emit_inputs, "join-emit",
                                           emit, checked_count);
}

} // namespace rund::compute
