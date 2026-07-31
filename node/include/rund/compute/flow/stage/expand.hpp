#ifndef RUND_COMPUTE_FLOW_STAGE_MEMBERS
#include <rund/compute/flow/stage.hpp>
#else
private:
  template <class Side, class CountFn, class EmitFn>
  [[nodiscard]] auto expand_impl(const MaxItems bound, const Side &side,
                                 CountFn &&count_function,
                                 EmitFn &&emit_function) const {
    using Count = CountFor<T>;
    using SourceCount = detail::ResidentCountT<T, Card>;
    constexpr bool HasSide =
        !std::same_as<std::remove_cvref_t<Side>, detail::NoSide>;
    if constexpr (HasSide) {
      if (state_ != side.state_) {
        detail::flow_pick(state_, 0u);
      }
    }
    auto emit_expressions = detail::make_expr();
    Expr<T> emit_value{
        detail::flow_expression_input<T>(state_, emit_expressions, value_, 0u)};
    auto emit = [&] {
      if constexpr (HasSide) {
        Expr<T> emit_side{detail::flow_expression_input<T>(
            state_, emit_expressions, side.value_, 1u)};
        Expr<T> emit_index{detail::flow_expression_input<T>(
            state_, emit_expressions, value_, 2u)};
        return detail::element(emit_function, emit_value, emit_side,
                               emit_index);
      } else {
        Expr<T> emit_index{detail::flow_expression_input<T>(
            state_, emit_expressions, value_, 1u)};
        return detail::element(emit_function, emit_value, emit_index);
      }
    }();
    using Emit = std::remove_cvref_t<decltype(emit)>;
    static_assert(detail::ComputeExpr<Emit> || detail::is_expr_record<Emit>,
                  "compute expand emit must return an expression or record");
    if constexpr (detail::ComputeExpr<Emit>) {
      static_assert(sizeof(detail::ExprValueT<Emit>) == sizeof(T),
                    "compute expand emit must preserve scalar width");
    } else {
      static_assert(
          detail::ExprRecordStage<Emit,
                                  stage::Bounded<Count>>::template accepts<T>,
          "all expand record fields must preserve scalar width");
    }
    const std::size_t source_count = detail::flow_value_count(state_, value_);
    constexpr std::size_t max_value =
        static_cast<std::size_t>(std::numeric_limits<T>::max());
    if (bound.value == 0u || bound.value > max_value ||
        source_count > std::numeric_limits<std::uint32_t>::max() ||
        source_count > std::numeric_limits<std::size_t>::max() / bound.value ||
        source_count * bound.value >
            std::numeric_limits<std::uint32_t>::max()) {
      detail::flow_reject(state_, Reason::ExpandCapacity);
      return detail::bounded_emit_reject<Emit, Count>(state_);
    }
    const std::size_t capacity = source_count * bound.value;
    if (source_count == 0u) {
      const std::uint32_t output =
          detail::flow_retype(state_, value_, detail::type<T>());
      const std::uint32_t zero_input =
          detail::flow_index(state_, detail::type<Count>(), 1u);
      const std::uint32_t count = detail::flow_unary_value(
          state_, zero_input, detail::Primitive::Reduce, detail::type<Count>(),
          1u, {.mode = static_cast<std::uint32_t>(Reduce::Sum)});
      if constexpr (HasSide) {
        const std::array empty_inputs{output, output, output};
        return detail::bounded_emit<Emit, Count>(state_, empty_inputs,
                                                 "expand-emit", emit, count);
      } else {
        const std::array empty_inputs{output, output};
        return detail::bounded_emit<Emit, Count>(state_, empty_inputs,
                                                 "expand-emit", emit, count);
      }
    }
    const T limit = static_cast<T>(bound.value);
    const auto counts = [&] {
      if constexpr (HasSide) {
        return combine("expand-count", side,
                       std::forward<CountFn>(count_function));
      } else {
        return map("expand-count", std::forward<CountFn>(count_function));
      }
    }();
    static_assert(std::same_as<typename decltype(counts)::Value, T>,
                  "compute expand count must preserve the integer value type");
    const auto invalid = counts.map(
        "expand-invalid",
        capture(
            [](auto count, auto maximum) {
              if constexpr (std::is_signed_v<T>) {
                return select((count > maximum) || (count < T{0}), T{1}, T{0});
              } else {
                return select(count > maximum, T{1}, T{0});
              }
            },
            limit));
    const auto invalid_value = invalid.reduce(Reduce::Max);
    const StageRef<Count, stage::Scalar> invalid_count{
        state_, detail::flow_retype(state_, invalid_value.value_,
                                    detail::type<Count>())};
    const auto validation_count = invalid_count.map(
        "expand-validation-count", [](auto value) { return value + Count{1}; });
    const StageRef<Count, stage::Exact> validation_input{
        state_, detail::flow_index(state_, detail::type<Count>(), 1u)};
    const std::array validation_inputs{validation_input.value_,
                                       validation_count.value_};
    const StageRef<Count, stage::Scalar> validation{
        state_, detail::flow_binary_values(
                    state_, detail::Primitive::Reduce, validation_inputs,
                    detail::type<Count>(), 1u,
                    {.mode = static_cast<std::uint32_t>(Reduce::Sum)})};

    StageRef<Count, stage::Exact> unsigned_counts{
        state_,
        detail::flow_retype(state_, counts.value_, detail::type<Count>())};
    if constexpr (detail::is_bounded_stage<Card>) {
      const StageRef<SourceCount, stage::Exact> source_slots{
          state_, detail::flow_index(state_, detail::type<SourceCount>(),
                                     source_count)};
      const StageRef<SourceCount, stage::Scalar> active_count{state_, count_};
      const auto active = source_slots.combine(
          "expand-active", active_count, [](auto index, auto count) {
            return detail::count_mask<Count>(index < count);
          });
      unsigned_counts = unsigned_counts.combine(
          "expand-active-count", active, [](auto count, auto selected) {
            return select(selected != Count{0}, count, Count{0});
          });
    }
    const StageRef<Count, stage::Exact> slots{
        state_, detail::flow_index(state_, detail::type<Count>(), capacity)};
    const auto local = slots.map(
        "expand-local",
        capture(
            [](auto slot, auto width) { return slot - (slot / width) * width; },
            static_cast<Count>(bound.value)));
    const StageRef<std::uint32_t, stage::Exact> source_slots{
        state_, detail::flow_index(state_, detail::Type::U32, capacity)};
    const auto sources = source_slots.map(
        "expand-source",
        capture([](auto slot, auto width) { return slot / width; },
                static_cast<std::uint32_t>(bound.value)));
    const std::array count_inputs{unsigned_counts.value_, sources.value_};
    const StageRef<Count, stage::Exact> slot_counts{
        state_, detail::flow_binary_values(state_, detail::Primitive::Gather,
                                           count_inputs, detail::type<Count>(),
                                           capacity, {})};
    const auto selected = slot_counts.combine(
        "expand-selected", local, [](auto count, auto index) {
          return select(index < count, Count{1}, Count{0});
        });
    const auto rejected = selected.map("expand-rejected", [](auto value) {
      return select(value != Count{0}, Count{0}, Count{1});
    });
    const auto selected_count = selected.count();
    const auto checked_count = selected_count.combine(
        "expand-checked-count", validation,
        [](auto count, auto check) { return count + check; });

    const std::array source_inputs{rejected.value_, sources.value_};
    const StageRef<std::uint32_t, stage::Exact> packed_sources{
        state_, detail::flow_binary_values(state_, detail::Primitive::Partition,
                                           source_inputs, detail::Type::U32,
                                           capacity, {})};
    const std::array local_inputs{rejected.value_, local.value_};
    const StageRef<Count, stage::Exact> packed_local{
        state_, detail::flow_binary_values(state_, detail::Primitive::Partition,
                                           local_inputs, detail::type<Count>(),
                                           capacity, {})};
    const std::array value_inputs{value_, packed_sources.value_};
    const StageRef<T, stage::Exact> packed_values{
        state_, detail::flow_binary_values(state_, detail::Primitive::Gather,
                                           value_inputs, detail::type<T>(),
                                           capacity, {})};
    const StageRef<T, stage::Exact> local_values{
        state_,
        detail::flow_retype(state_, packed_local.value_, detail::type<T>())};
    if constexpr (HasSide) {
      const std::uint32_t zeros = detail::flow_zero(state_, capacity);
      const std::array side_inputs{side.value_, zeros};
      const std::uint32_t side_values = detail::flow_binary_values(
          state_, detail::Primitive::Gather, side_inputs, detail::type<T>(),
          capacity, {});
      const std::array emit_inputs{packed_values.value_, side_values,
                                   local_values.value_};
      return detail::bounded_emit<Emit, Count>(
          state_, emit_inputs, "expand-emit", emit, checked_count.value_);
    } else {
      const std::array emit_inputs{packed_values.value_, local_values.value_};
      return detail::bounded_emit<Emit, Count>(
          state_, emit_inputs, "expand-emit", emit, checked_count.value_);
    }
  }
#endif
