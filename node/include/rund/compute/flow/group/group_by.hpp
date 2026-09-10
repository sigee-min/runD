#pragma once

#include <rund/compute/flow/group/groups.hpp>

namespace rund::compute {

template <class T, class Card>
template <class Fn>
[[nodiscard]] auto StageRef<T, Card>::group_by(Fn &&function) const
  requires((std::same_as<Card, stage::Exact> ||
            detail::is_bounded_stage<Card>) &&
           detail::IntegerValue<T>)
{
  auto expressions = detail::make_expr();
  Expr<T> value{
      detail::flow_expression_input<T>(state_, expressions, value_, 0u)};
  auto key_expression = detail::element(function, value);
  static_assert(detail::ComputeExpr<decltype(key_expression)>,
                "compute group key must be a compute expression");
  using Key = detail::ExprValueT<decltype(key_expression)>;
  static_assert(detail::IntegerValue<Key> && sizeof(Key) == sizeof(T),
                "compute group key must preserve scalar width");
  using Count = std::uint32_t;
  const std::size_t count = detail::flow_value_count(state_, value_);
  if (count > std::numeric_limits<Count>::max()) {
    detail::flow_reject(state_, Reason::GroupCapacity);
    return Groups<Key, T, Count>{state_, value_, value_, value_,
                                 value_, value_, value_};
  }
  const std::array key_inputs{value_};
  const std::uint32_t keys = detail::flow_map_value(
      state_, key_inputs, "group-key", key_expression.ref_);
  using SourceCount = CountFor<T>;
  using ResidentCount = detail::ResidentCountT<T, Card>;
  const std::uint32_t source_count = [&] {
    if constexpr (detail::is_bounded_stage<Card>) {
      if constexpr (std::same_as<SourceCount, ResidentCount>) {
        return count_;
      } else {
        const StageRef<ResidentCount, stage::Exact> positions{
            state_,
            detail::flow_index(state_, detail::type<ResidentCount>(), count)};
        const StageRef<ResidentCount, stage::Scalar> logical{state_, count_};
        const auto active = positions.combine(
            "group-source-active", logical, [](auto index, auto size) {
              return detail::count_mask<SourceCount>(index < size);
            });
        return active.reduce(Reduce::Sum).value_;
      }
    } else {
      const std::uint32_t count_input =
          detail::flow_index(state_, detail::type<SourceCount>(), 1u);
      auto expressions = detail::make_expr();
      Expr<SourceCount> zero{detail::flow_expression_input<SourceCount>(
          state_, expressions, count_input, 0u)};
      const Expr<SourceCount> logical = zero + static_cast<SourceCount>(count);
      const std::array inputs{count_input};
      return detail::flow_map_value(state_, inputs, "group-source-count",
                                    logical.ref_);
    }
  }();
  if (count == 0u) {
    auto head_expressions = detail::make_expr();
    Expr<T> empty_value{
        detail::flow_expression_input<T>(state_, head_expressions, value_, 0u)};
    const Expr<std::uint32_t> empty_head = mask(empty_value != empty_value);
    const std::array head_inputs{value_};
    const std::uint32_t heads = detail::flow_map_value(
        state_, head_inputs, "group-empty-head", empty_head.ref_);
    const std::uint32_t zero_input =
        detail::flow_index(state_, detail::Type::U32, 1u);
    const std::uint32_t group_count = detail::flow_unary_value(
        state_, zero_input, detail::Primitive::Reduce, detail::Type::U32, 1u,
        {.mode = static_cast<std::uint32_t>(Reduce::Sum)});
    return Groups<Key, T>{state_, keys,        value_,      heads,
                          value_, group_count, source_count};
  }
  const std::uint32_t order = [&] {
    if constexpr (detail::is_bounded_stage<Card>) {
      return detail::flow_bounded_sort_value(state_, keys, count_, true);
    } else {
      return detail::flow_unary_value(state_, keys, detail::Primitive::Argsort,
                                      detail::Type::U32, count, {});
    }
  }();
  const std::array sorted_key_inputs{keys, order};
  const std::uint32_t sorted_keys = detail::flow_binary_values(
      state_, detail::Primitive::Gather, sorted_key_inputs, detail::type<Key>(),
      count, {});
  const std::array sorted_value_inputs{value_, order};
  const std::uint32_t sorted_values = detail::flow_binary_values(
      state_, detail::Primitive::Gather, sorted_value_inputs, detail::type<T>(),
      count, {});

  const StageRef<std::uint32_t, stage::Exact> slots{
      state_, detail::flow_index(state_, detail::Type::U32, count)};
  const auto previous = slots.map("group-previous", [](auto index) {
    return select(index == 0u, 0u, index - 1u);
  });
  const std::array previous_inputs{sorted_keys, previous.value_};
  const std::uint32_t previous_keys = detail::flow_binary_values(
      state_, detail::Primitive::Gather, previous_inputs, detail::type<Key>(),
      count, {});

  auto head_expressions = detail::make_expr();
  Expr<Key> current{detail::flow_expression_input<Key>(state_, head_expressions,
                                                       sorted_keys, 0u)};
  Expr<Key> prior{detail::flow_expression_input<Key>(state_, head_expressions,
                                                     previous_keys, 1u)};
  Expr<Key> index{detail::index(head_expressions, detail::type<Key>())};
  const auto boundary = index == Key{0} || current != prior;
  const Expr<Key> typed_boundary = select(boundary, Key{1}, Key{0});
  const std::array head_inputs{sorted_keys, previous_keys};
  const StageRef<Key, stage::Exact> boundaries{
      state_, detail::flow_map_value(state_, head_inputs, "group-boundary",
                                     typed_boundary.ref_)};
  const StageRef<SourceCount, stage::Exact> source_slots{
      state_, detail::flow_index(state_, detail::type<SourceCount>(), count)};
  const StageRef<SourceCount, stage::Scalar> logical_count{state_,
                                                           source_count};
  const auto active = source_slots.combine(
      "group-active", logical_count, [](auto position, auto logical) {
        return select(position < logical, SourceCount{1}, SourceCount{0});
      });
  const StageRef<Key, stage::Exact> typed_active{
      state_, detail::flow_retype(state_, active.value_, detail::type<Key>())};
  const auto typed_heads = boundaries.combine(
      "group-typed-head", typed_active,
      [](auto head, auto selected) { return head & selected; });
  const auto head_values = typed_heads.map(
      "group-head", [](auto head) { return mask(head != Key{0}); });
  const std::uint32_t heads = head_values.value_;
  const std::uint32_t marks =
      detail::flow_retype(state_, typed_heads.value_, detail::type<T>());

  const StageRef<Count, stage::Exact> head_stage{state_, heads};
  const StageRef<Count, stage::Bounded<Count>> head_indices =
      head_stage.compact({.capacity = count});
  const StageRef<Count, stage::Scalar> resident_group_count{
      state_, head_indices.count_};
  const auto active_head_index =
      slots.combine("group-head-index-active", resident_group_count,
                    [](auto position, auto logical) {
                      return select(position < logical, Count{1}, Count{0});
                    });
  const StageRef<Count, stage::Exact> physical_head_indices{
      state_, head_indices.value_};
  const auto safe_head_indices = physical_head_indices.combine(
      "group-head-index-safe", active_head_index, [](auto index, auto active) {
        return select(active != Count{0}, index, Count{0});
      });
  const std::array packed_key_inputs{sorted_keys, safe_head_indices.value_};
  const std::uint32_t packed_keys = detail::flow_binary_values(
      state_, detail::Primitive::Gather, packed_key_inputs, detail::type<Key>(),
      count, {});
  return Groups<Key, T>{state_, packed_keys,         sorted_values, heads,
                        marks,  head_indices.count_, source_count};
}

} // namespace rund::compute
