#ifndef RUND_COMPUTE_FLOW_STAGE_MEMBERS
#include <rund/compute/flow/stage.hpp>
#else
public:
  [[nodiscard]] StageRef window(const WindowSpec options) const
    requires(std::same_as<Card, stage::Exact> || detail::is_bounded_stage<Card>)
  {
    if (options.edge != WindowEdge::Clamp && options.edge != WindowEdge::Clip) {
      detail::flow_reject(state_, Reason::WindowEdgeUnsupported);
    }
    if (options.radius == 0u) {
      detail::flow_reject(state_, Reason::StencilRadiusInvalid);
      return *this;
    }
    const std::size_t size = detail::flow_value_count(state_, value_);
    if (size == 0u) {
      if (options.op != Window::Sum) {
        detail::flow_reject(state_, Reason::StencilCountZero);
        return *this;
      }
      return {state_,
              detail::flow_unary_value(
                  state_, value_, detail::Primitive::Stencil, detail::type<T>(),
                  0u,
                  {.first = options.radius,
                   .mode = static_cast<std::uint32_t>(options.op),
                   .flag = options.edge == WindowEdge::Clip}),
              count_};
    }
    if (options.radius > size) {
      detail::flow_reject(state_, Reason::StencilRadiusInvalid);
      return *this;
    }
    if ((detail::is_bounded_stage<Card> || options.edge == WindowEdge::Clip) &&
        size > std::numeric_limits<std::uint32_t>::max()) {
      detail::flow_reject(state_, Reason::StencilCountOverflow);
      return *this;
    }
    if constexpr (std::same_as<Card, stage::Exact>) {
      if (options.edge == WindowEdge::Clamp) {
        return {state_,
                detail::flow_unary_value(
                    state_, value_, detail::Primitive::Stencil,
                    detail::type<T>(), size,
                    {.first = options.radius,
                     .mode = static_cast<std::uint32_t>(options.op)}),
                count_};
      }
    }
    {
      using Count = detail::ResidentCountT<T, Card>;
      using Active =
          std::conditional_t<sizeof(T) == sizeof(Count), Count, CountFor<T>>;
      const std::uint32_t zeros = detail::flow_zero(state_, size);
      const StageRef<Count, stage::Exact> counts = [&] {
        if constexpr (detail::is_bounded_stage<Card>) {
          const std::array count_inputs{count_, zeros};
          return StageRef<Count, stage::Exact>{
              state_, detail::flow_binary_values(
                          state_, detail::Primitive::Gather, count_inputs,
                          detail::type<Count>(), size, {})};
        } else {
          const StageRef<Count, stage::Exact> origin{
              state_, detail::flow_index(state_, detail::type<Count>(), 1u)};
          const auto logical = origin.map(
              "window-logical-count",
              capture([](auto value, auto count) { return value + count; },
                      static_cast<Count>(size)));
          const std::array count_inputs{logical.value_, zeros};
          return StageRef<Count, stage::Exact>{
              state_, detail::flow_binary_values(
                          state_, detail::Primitive::Gather, count_inputs,
                          detail::type<Count>(), size, {})};
        }
      }();
      const StageRef<Count, stage::Exact> positions{
          state_, detail::flow_index(state_, detail::type<Count>(), size)};
      const auto last_mark = positions.combine(
          "window-last-mark", counts, [](auto index, auto count) {
            return detail::count_mask<Active>(count != Count{0} &&
                                              index + Count{1} == count);
          });
      const StageRef<T, stage::Exact> typed_last_mark{
          state_, detail::flow_retype_like(state_, last_mark.value_, value_)};
      const T zero = [] {
        if constexpr (detail::FixedValue<T>) {
          return T::zero();
        } else {
          return T{0};
        }
      }();
      const StageRef<T, stage::Exact> selected_last = [&] {
        if constexpr (detail::FixedValue<T>) {
          return StageRef<T, stage::Exact>{
              state_,
              detail::flow_fixed_select_value(
                  state_, value_, typed_last_mark.value_,
                  detail::static_bits(zero), false, "window-last-select")};
        } else {
          return StageRef<T, stage::Exact>{state_, value_}.combine(
              "window-last-select", typed_last_mark,
              capture(
                  [](auto value, auto selected, auto zero_value) {
                    return select(selected != zero_value, value, zero_value);
                  },
                  zero));
        }
      }();
      const auto last_scalar = selected_last.reduce(Reduce::Sum);
      const std::array last_broadcast_inputs{last_scalar.value_, zeros};
      const StageRef<T, stage::Exact> last{
          state_, detail::flow_binary_values(state_, detail::Primitive::Gather,
                                             last_broadcast_inputs,
                                             detail::type<T>(), size, {})};
      const StageRef<std::uint32_t, stage::Exact> slots{
          state_, detail::flow_index(state_, detail::Type::U32, size)};
      StageRef<T, stage::Exact> output{state_, value_};
      const auto merge = [options](const StageRef<T, stage::Exact> &left,
                                   const StageRef<T, stage::Exact> &right) {
        if constexpr (detail::FixedValue<T>) {
          return StageRef<T, stage::Exact>{
              left.state_, detail::flow_fixed_merge_values(
                               left.state_, left.value_, right.value_,
                               options.op, "window-merge")};
        } else {
          if (options.op == Window::Sum) {
            return left.combine(
                "window-merge", right,
                [](auto value, auto neighbor) { return value + neighbor; });
          }
          if (options.op == Window::Min) {
            return left.combine(
                "window-merge", right, [](auto value, auto neighbor) {
                  return select(value < neighbor, value, neighbor);
                });
          }
          return left.combine(
              "window-merge", right, [](auto value, auto neighbor) {
                return select(value > neighbor, value, neighbor);
              });
        }
      };
      for (std::size_t distance = 1u; distance <= options.radius; ++distance) {
        const std::uint32_t step = static_cast<std::uint32_t>(distance);
        const std::uint32_t physical_last =
            static_cast<std::uint32_t>(size - 1u);
        const auto left_indices =
            slots.map("window-left-index",
                      capture(
                          [](auto index, auto offset) {
                            return select(index < offset, 0u, index - offset);
                          },
                          step));
        const auto right_indices = slots.map(
            "window-right-index",
            capture(
                [](auto index, auto offset, auto last_index) {
                  return select(offset > last_index, index - index + last_index,
                                select(index > last_index - offset, last_index,
                                       index + offset));
                },
                step, physical_last));
        const std::array left_inputs{value_, left_indices.value_};
        const StageRef<T, stage::Exact> left{
            state_, detail::flow_binary_values(
                        state_, detail::Primitive::Gather, left_inputs,
                        detail::type<T>(), size, {})};
        const std::array right_inputs{value_, right_indices.value_};
        const StageRef<T, stage::Exact> right{
            state_, detail::flow_binary_values(
                        state_, detail::Primitive::Gather, right_inputs,
                        detail::type<T>(), size, {})};
        const Count typed_step = static_cast<Count>(distance);
        const auto right_active = positions.combine(
            "window-right-active", counts,
            capture(
                [](auto index, auto count, auto offset) {
                  return detail::count_mask<Active>(count > offset &&
                                                    index < count - offset);
                },
                typed_step));
        const StageRef<T, stage::Exact> active{
            state_,
            detail::flow_retype_like(state_, right_active.value_, value_)};
        if (options.edge == WindowEdge::Clip) {
          const T identity = [options] {
            if (options.op == Window::Min) {
              if constexpr (detail::FixedValue<T>) {
                return T::max();
              } else {
                return std::numeric_limits<T>::max();
              }
            }
            if (options.op == Window::Max) {
              if constexpr (detail::FixedValue<T>) {
                return T::min();
              } else {
                return std::numeric_limits<T>::lowest();
              }
            }
            if constexpr (detail::FixedValue<T>) {
              return T::zero();
            } else {
              return T{0};
            }
          }();
          const auto left_active = positions.map(
              "window-left-active",
              capture(
                  [](auto index, auto offset) {
                    return detail::count_mask<Active>(index >= offset);
                  },
                  typed_step));
          const StageRef<T, stage::Exact> typed_left_active{
              state_,
              detail::flow_retype_like(state_, left_active.value_, value_)};
          const StageRef<T, stage::Exact> selected_left = [&] {
            if constexpr (detail::FixedValue<T>) {
              return StageRef<T, stage::Exact>{
                  state_, detail::flow_fixed_select_value(
                              state_, left.value_, typed_left_active.value_,
                              detail::static_bits(identity), false,
                              "window-left-select")};
            } else {
              return left.combine("window-left-select", typed_left_active,
                                  capture(
                                      [](auto value, auto selected,
                                         auto identity_value, auto zero_value) {
                                        return select(selected != zero_value,
                                                      value, identity_value);
                                      },
                                      identity, zero));
            }
          }();
          const StageRef<T, stage::Exact> selected_right = [&] {
            if constexpr (detail::FixedValue<T>) {
              return StageRef<T, stage::Exact>{
                  state_, detail::flow_fixed_select_value(
                              state_, right.value_, active.value_,
                              detail::static_bits(identity), false,
                              "window-right-select")};
            } else {
              return right.combine(
                  "window-right-select", active,
                  capture(
                      [](auto value, auto selected, auto identity_value,
                         auto zero_value) {
                        return select(selected != zero_value, value,
                                      identity_value);
                      },
                      identity, zero));
            }
          }();
          output = merge(output, selected_left);
          output = merge(output, selected_right);
          continue;
        }
        const StageRef<T, stage::Exact> selected_right = [&] {
          if constexpr (detail::FixedValue<T>) {
            return StageRef<T, stage::Exact>{
                state_,
                detail::flow_fixed_select_value(
                    state_, right.value_, active.value_,
                    detail::static_bits(zero), false, "window-right-select")};
          } else {
            return right.combine(
                "window-right-select", active,
                capture(
                    [](auto value, auto selected, auto zero_value) {
                      return select(selected != zero_value, value, zero_value);
                    },
                    zero));
          }
        }();
        const StageRef<T, stage::Exact> rejected_right = [&] {
          if constexpr (detail::FixedValue<T>) {
            return StageRef<T, stage::Exact>{
                state_,
                detail::flow_fixed_select_value(
                    state_, last.value_, active.value_,
                    detail::static_bits(zero), true, "window-right-clamp")};
          } else {
            return last.combine(
                "window-right-clamp", active,
                capture(
                    [](auto value, auto selected, auto zero_value) {
                      return select(selected == zero_value, value, zero_value);
                    },
                    zero));
          }
        }();
        output = merge(output, left);
        const StageRef<T, stage::Exact> right_value = [&] {
          if constexpr (detail::FixedValue<T>) {
            return StageRef<T, stage::Exact>{
                state_,
                detail::flow_fixed_merge_values(
                    state_, selected_right.value_, rejected_right.value_,
                    Window::Sum, "window-right-value")};
          } else {
            return selected_right.combine(
                "window-right-value", rejected_right,
                [](auto selected, auto clamped) { return selected + clamped; });
          }
        }();
        output = merge(output, right_value);
      }
      return {state_, output.value_, count_};
    }
  }
#endif
