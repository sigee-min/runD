#pragma once

#include <rund/compute/flow/stage.hpp>

namespace rund::compute {
template <class T, class SourceCount> class GroupValuesRef final {
public:
  using Count = std::uint32_t;
  template <class Fn>
  [[nodiscard]] auto map(const std::string_view name, Fn &&function) const {
    const StageRef<T, stage::Exact> values{state_, values_};
    auto mapped = values.map(name, std::forward<Fn>(function));
    using U = typename decltype(mapped)::Value;
    const std::uint32_t mapped_marks = [&] {
      if constexpr (std::same_as<U, std::uint32_t>) {
        return heads_;
      }
      return detail::flow_retype_like(state_, marks_, mapped.value_);
    }();
    return GroupValuesRef<U, SourceCount>{
        state_, mapped.value_, heads_, mapped_marks, count_, source_count_};
  }
  [[nodiscard]] GroupValuesRef scan(const Scan operation) const {
    const auto values = StageRef<T, stage::Exact>{state_, values_}.combine(
        "group-active-scan", active("group-scan-active"),
        [](auto value, auto selected) {
          return select(selected != T{0}, value, T{0});
        });
    const std::array inputs{values.value_, heads_};
    const std::uint32_t scanned = detail::flow_binary_values(
        state_, detail::Primitive::SegmentedScan, inputs, detail::type<T>(),
        detail::flow_value_count(state_, values_),
        {.mode = static_cast<std::uint32_t>(operation)});
    return {state_, scanned, heads_, marks_, count_, source_count_};
  }
  [[nodiscard]] GroupValuesRef window(const WindowSpec options) const {
    const std::size_t size = detail::flow_value_count(state_, values_);
    if (options.edge != WindowEdge::Clip || options.radius == 0u ||
        options.radius > size ||
        size > std::numeric_limits<std::uint32_t>::max()) {
      detail::flow_reject(state_, Reason::GroupWindowInvalid);
      return *this;
    }
    const StageRef<std::uint32_t, stage::Exact> slots{
        state_, detail::flow_index(state_, detail::Type::U32, size)};
    const StageRef<T, stage::Exact> segments{
        state_, detail::flow_scan_value(state_, marks_, Scan::InclusiveSum)};
    const StageRef<T, stage::Exact> validity_slots{
        state_, detail::flow_index(state_, detail::type<T>(), size)};
    const auto active_values = active("group-window-active");
    StageRef<T, stage::Exact> output{state_, values_};

    const auto append = [&](const bool right, const std::size_t distance) {
      const std::uint32_t step = static_cast<std::uint32_t>(distance);
      const std::uint32_t last = static_cast<std::uint32_t>(size - 1u);
      const auto indices =
          right ? slots.map("group-window-index",
                            capture(
                                [](auto index, auto offset, auto last_index) {
                                  return select(index > last_index - offset,
                                                last_index, index + offset);
                                },
                                step, last))
                : slots.map("group-window-index",
                            capture(
                                [](auto index, auto offset, auto) {
                                  return select(index < offset, 0u,
                                                index - offset);
                                },
                                step, last));
      const T offset = static_cast<T>(step);
      const T end = static_cast<T>(last);
      const auto valid =
          right ? validity_slots.map(
                      "group-window-valid",
                      capture(
                          [](auto index, auto distance, auto last_index) {
                            return select(index <= last_index - distance, T{1},
                                          T{0});
                          },
                          offset, end))
                : validity_slots.map("group-window-valid",
                                     capture(
                                         [](auto index, auto distance, auto) {
                                           return select(index >= distance,
                                                         T{1}, T{0});
                                         },
                                         offset, end));
      const std::array value_inputs{values_, indices.value_};
      const StageRef<T, stage::Exact> neighbor{
          state_, detail::flow_binary_values(state_, detail::Primitive::Gather,
                                             value_inputs, detail::type<T>(),
                                             size, {})};
      const std::array segment_inputs{segments.value_, indices.value_};
      const StageRef<T, stage::Exact> neighbor_segments{
          state_, detail::flow_binary_values(state_, detail::Primitive::Gather,
                                             segment_inputs, detail::type<T>(),
                                             size, {})};
      const auto same =
          segments.combine("group-window-segment", neighbor_segments,
                           [](auto current, auto candidate) {
                             return select(current == candidate, T{1}, T{0});
                           });
      const auto selected = same.combine(
          "group-window-selected", valid,
          [](auto segment, auto in_range) { return segment & in_range; });
      const std::array active_inputs{active_values.value_, indices.value_};
      const StageRef<T, stage::Exact> neighbor_active{
          state_, detail::flow_binary_values(state_, detail::Primitive::Gather,
                                             active_inputs, detail::type<T>(),
                                             size, {})};
      const auto active_selected = selected.combine(
          "group-window-active-selected", neighbor_active,
          [](auto selected, auto active) { return selected & active; });
      const T identity = [options] {
        if (options.op == Window::Min) {
          return std::numeric_limits<T>::max();
        }
        if (options.op == Window::Max) {
          return std::numeric_limits<T>::min();
        }
        return T{0};
      }();
      const auto candidate =
          neighbor.combine("group-window-mask", active_selected,
                           capture(
                               [](auto value, auto selected, auto empty) {
                                 return select(selected != T{0}, value, empty);
                               },
                               identity));
      if (options.op == Window::Sum) {
        return output.combine("group-window-merge", candidate,
                              [](auto value, auto candidate_value) {
                                return value + candidate_value;
                              });
      }
      if (options.op == Window::Min) {
        return output.combine("group-window-merge", candidate,
                              [](auto value, auto candidate_value) {
                                constexpr T empty =
                                    std::numeric_limits<T>::max();
                                return select(candidate_value == empty, value,
                                              select(value < candidate_value,
                                                     value, candidate_value));
                              });
      }
      return output.combine("group-window-merge", candidate,
                            [](auto value, auto candidate_value) {
                              constexpr T empty = std::numeric_limits<T>::min();
                              return select(candidate_value == empty, value,
                                            select(value > candidate_value,
                                                   value, candidate_value));
                            });
    };
    for (std::size_t distance = 1u;
         distance <= options.radius && distance < size; ++distance) {
      output = append(false, distance);
      output = append(true, distance);
    }
    return {state_, output.value_, heads_, marks_, count_, source_count_};
  }
  [[nodiscard]] StageRef<T, stage::Bounded<SourceCount>> ordered() const {
    return {state_, values_, source_count_};
  }
  [[nodiscard]] StageRef<T, stage::Bounded<Count>>
  reduce(const Reduce operation = Reduce::Sum) const {
    if (detail::flow_value_count(state_, values_) == 0u) {
      return {state_, values_, copy_count("group-value-count")};
    }
    const auto values = masked(operation);
    const std::array inputs{values.value_, heads_};
    const std::uint32_t reduced = detail::flow_binary_values(
        state_, detail::Primitive::SegmentedReduce, inputs, detail::type<T>(),
        detail::flow_value_count(state_, values_),
        {.mode = static_cast<std::uint32_t>(operation)});
    return {state_, reduced, copy_count("group-value-count")};
  }

private:
  template <class, class, class> friend class Groups;
  template <class, class> friend class GroupValuesRef;
  GroupValuesRef(std::shared_ptr<detail::FlowState> state,
                 const std::uint32_t values, const std::uint32_t heads,
                 const std::uint32_t marks, const std::uint32_t count,
                 const std::uint32_t source_count)
      : state_(std::move(state)), values_(values), heads_(heads), marks_(marks),
        count_(count), source_count_(source_count) {}
  [[nodiscard]] StageRef<T, stage::Exact>
  active(const std::string_view name) const {
    const std::size_t size = detail::flow_value_count(state_, values_);
    if (size == 0u) {
      return {state_, values_};
    }
    const StageRef<SourceCount, stage::Exact> positions{
        state_, detail::flow_index(state_, detail::type<SourceCount>(), size)};
    const StageRef<SourceCount, stage::Scalar> count{state_, source_count_};
    const auto selected =
        positions.combine(name, count, [](auto index, auto logical) {
          return select(index < logical, SourceCount{1}, SourceCount{0});
        });
    return {state_, detail::flow_retype_like(state_, selected.value_, values_)};
  }
  [[nodiscard]] StageRef<T, stage::Exact> masked(const Reduce operation) const {
    if (detail::flow_value_count(state_, values_) == 0u) {
      return {state_, values_};
    }
    const T identity = [operation] {
      if (operation == Reduce::Min) {
        return std::numeric_limits<T>::max();
      }
      if (operation == Reduce::Max) {
        return std::numeric_limits<T>::min();
      }
      return T{0};
    }();
    return StageRef<T, stage::Exact>{state_, values_}.combine(
        "group-active-value", active("group-active"),
        capture(
            [](auto value, auto selected, auto empty) {
              return select(selected != T{0}, value, empty);
            },
            identity));
  }
  [[nodiscard]] std::uint32_t copy_count(const std::string_view name) const {
    auto expressions = detail::make_expr();
    Expr<Count> count{
        detail::flow_expression_input<Count>(state_, expressions, count_, 0u)};
    const std::array inputs{count_};
    return detail::flow_map_value(state_, inputs, name, count.ref_);
  }
  std::shared_ptr<detail::FlowState> state_;
  std::uint32_t values_{};
  std::uint32_t heads_{};
  std::uint32_t marks_{};
  std::uint32_t count_{};
  std::uint32_t source_count_{};
};

template <class Key, class T, class C> class Groups final {
public:
  using Value = T;
  using KeyType = Key;
  using Count = C;
  static_assert(std::same_as<Count, std::uint32_t>,
                "compute grouped count must be uint32_t");

  template <class Fn> [[nodiscard]] decltype(auto) aggregate(Fn &&function) {
    return std::forward<Fn>(function)(*this);
  }
  [[nodiscard]] StageRef<Key, stage::Bounded<Count>> key() const {
    return {state_, keys_, copy_count("group-key-count")};
  }
  [[nodiscard]] StageRef<Count, stage::Bounded<Count>> count() const {
    const auto expressions = detail::make_expr();
    Expr<Count> head{
        detail::flow_expression_input<Count>(state_, expressions, heads_, 0u)};
    const Expr<Count> unit = head - head + Count{1};
    const std::array unit_inputs{heads_};
    const std::uint32_t raw_units =
        detail::flow_map_value(state_, unit_inputs, "group-unit", unit.ref_);
    const std::size_t size = detail::flow_value_count(state_, values_);
    if (size == 0u) {
      return {state_, raw_units, copy_count("group-size-count")};
    }
    using SourceCount = CountFor<T>;
    const StageRef<SourceCount, stage::Exact> positions{
        state_, detail::flow_index(state_, detail::type<SourceCount>(), size)};
    const StageRef<SourceCount, stage::Scalar> source_count{state_,
                                                            source_count_};
    const auto active = positions.combine(
        "group-size-active", source_count,
        [](auto index, auto logical) { return mask(index < logical); });
    const auto units = StageRef<Count, stage::Exact>{state_, raw_units}.combine(
        "group-active-unit", active, [](auto value, auto selected) {
          return select(selected != Count{0}, value, Count{0});
        });
    const std::array reduce_inputs{units.value_, heads_};
    const std::uint32_t counts = detail::flow_binary_values(
        state_, detail::Primitive::SegmentedReduce, reduce_inputs,
        detail::type<Count>(), detail::flow_value_count(state_, heads_),
        {.mode = static_cast<std::uint32_t>(Reduce::Sum)});
    return {state_, counts, copy_count("group-size-count")};
  }
  [[nodiscard]] GroupValuesRef<T, CountFor<T>> values() const {
    return {state_, values_, heads_, marks_, count_, source_count_};
  }

private:
  template <class, class> friend class StageRef;
  Groups(std::shared_ptr<detail::FlowState> state, const std::uint32_t keys,
         const std::uint32_t values, const std::uint32_t heads,
         const std::uint32_t marks, const std::uint32_t count,
         const std::uint32_t source_count)
      : state_(std::move(state)), keys_(keys), values_(values), heads_(heads),
        marks_(marks), count_(count), source_count_(source_count) {}
  [[nodiscard]] std::uint32_t copy_count(const std::string_view name) const {
    auto expressions = detail::make_expr();
    Expr<Count> count{
        detail::flow_expression_input<Count>(state_, expressions, count_, 0u)};
    const std::array inputs{count_};
    return detail::flow_map_value(state_, inputs, name, count.ref_);
  }
  std::shared_ptr<detail::FlowState> state_;
  std::uint32_t keys_{};
  std::uint32_t values_{};
  std::uint32_t heads_{};
  std::uint32_t marks_{};
  std::uint32_t count_{};
  std::uint32_t source_count_{};
};

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
