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

} // namespace rund::compute
