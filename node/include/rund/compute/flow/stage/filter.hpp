#ifndef RUND_COMPUTE_FLOW_STAGE_MEMBERS
#include <rund/compute/flow/stage.hpp>
#else
  template <class Fn>
  [[nodiscard]] auto filter(Fn &&function) const
    requires(std::same_as<Card, stage::Exact> || detail::is_bounded_stage<Card>)
  {
    auto expressions = detail::make_expr();
    Expr<T> argument{
        detail::flow_expression_input<T>(state_, expressions, value_, 0u)};
    auto predicate = detail::element(function, argument);
    static_assert(
        std::is_same_v<std::remove_cvref_t<decltype(predicate)>, Predicate<T>>,
        "compute branch filter must return a predicate");
    using Count = detail::ResidentCountT<T, Card>;
    const Expr<Count> selected = detail::count_mask<Count>(predicate);
    const Expr<Count> rejected = detail::count_mask<Count>(!predicate);
    const detail::BoundedIds result = [&] {
      if constexpr (std::same_as<Card, stage::Exact>) {
        return detail::flow_filter_value(state_, value_, selected.ref_,
                                         rejected.ref_);
      } else {
        const std::size_t capacity = detail::flow_value_count(state_, value_);
        const std::uint32_t selected_mask = detail::flow_map_value(
            state_, std::array{value_}, "filter-selected", selected.ref_);
        const StageRef<Count, stage::Exact> positions{
            state_,
            detail::flow_index(state_, detail::type<Count>(), capacity)};
        const std::uint32_t zeros = detail::flow_zero(state_, capacity);
        const std::array count_inputs{count_, zeros};
        const StageRef<Count, stage::Exact> counts{
            state_, detail::flow_binary_values(
                        state_, detail::Primitive::Gather, count_inputs,
                        detail::type<Count>(), capacity, {})};
        const auto active = positions.combine(
            "filter-active", counts, [](auto index, auto logical) {
              return select(index < logical, Count{1}, Count{0});
            });
        const StageRef<Count, stage::Exact> selected_stage{state_,
                                                           selected_mask};
        const auto live = selected_stage.combine(
            "filter-live", active,
            [](auto chosen, auto enabled) { return chosen & enabled; });
        const auto rejected = live.map("filter-rejected", [](auto chosen) {
          return select(chosen != Count{0}, Count{0}, Count{1});
        });
        return detail::flow_filter_masks(state_, value_, live.value_,
                                         rejected.value_);
      }
    }();
    return StageRef<T, stage::Bounded<Count>>{state_, result.values,
                                              result.count};
  }
  [[nodiscard]] StageRef scan(const Scan operation) const
    requires std::same_as<Card, stage::Exact>
  {
    return {state_, detail::flow_scan_value(state_, value_, operation)};
  }
  [[nodiscard]] StageRef scan(const Scan operation) const
    requires detail::is_bounded_stage<Card>
  {
    return {state_,
            detail::flow_bounded_scan_value(state_, value_, count_, operation),
            count_};
  }
  [[nodiscard]] StageRef<CountFor<T>, Card> indices() const
    requires std::same_as<Card, stage::Exact>
  {
    return {state_,
            detail::flow_index(state_, detail::type<CountFor<T>>(),
                               detail::flow_value_count(state_, value_)),
            count_};
  }
  [[nodiscard]] StageRef<detail::ResidentCountT<T, Card>, Card> indices() const
    requires detail::is_bounded_stage<Card>
  {
    using Count = detail::ResidentCountT<T, Card>;
    return {state_,
            detail::flow_index(state_, detail::type<Count>(),
                               detail::flow_value_count(state_, value_)),
            count_};
  }
  template <class CountFn, class EmitFn>
  [[nodiscard]] auto expand(const MaxItems bound, CountFn &&count_function,
                            EmitFn &&emit_function) const
    requires((std::same_as<Card, stage::Exact> ||
              detail::is_bounded_stage<Card>) &&
             detail::IntegerValue<T>)
  {
    return expand_impl(bound, detail::NoSide{},
                       std::forward<CountFn>(count_function),
                       std::forward<EmitFn>(emit_function));
  }
  template <class CountFn, class EmitFn>
  [[nodiscard]] auto
  expand(const MaxItems bound, const StageRef<T, stage::Scalar> &side,
         CountFn &&count_function, EmitFn &&emit_function) const
    requires((std::same_as<Card, stage::Exact> ||
              detail::is_bounded_stage<Card>) &&
             detail::IntegerValue<T>)
  {
    return expand_impl(bound, side, std::forward<CountFn>(count_function),
                       std::forward<EmitFn>(emit_function));
  }
#endif
