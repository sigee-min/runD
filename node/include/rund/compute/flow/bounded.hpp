#pragma once

#include <rund/compute/flow/deferred.hpp>

namespace rund::compute {
template <class R, class Count, class Inputs, class... A>
  requires detail::InputMode<Inputs>
class Flow<R(A...), stage::Bounded<Count>, Inputs> final
    : public detail::StagePipe<Flow<R(A...), stage::Bounded<Count>, Inputs>> {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;

private:
  friend class detail::StagePipe<Flow<R(A...), stage::Bounded<Count>, Inputs>>;
  template <class Fn> [[nodiscard]] auto pipe_stage(Fn &&function) && {
    auto next = std::forward<Fn>(function)(
        StageRef<R, stage::Bounded<Count>>{state_, values_, count_});
    using Next = std::remove_cvref_t<decltype(next)>;
    static_assert(detail::is_stage_ref<Next>,
                  "compute pipe must return a typed stage");
    if (state_ != detail::StageRefAccess::state(next)) {
      detail::flow_pick(state_, 0u);
    }
    using U = typename Next::Value;
    using Card = typename Next::Cardinality;
    if constexpr (detail::is_bounded_stage<Card>) {
      return Flow<U(A...), Card, Inputs>{std::move(state_), next.value_,
                                         next.count_};
    } else {
      detail::flow_pick(state_, next.value_);
      return Flow<U(A...), Card, Inputs>{std::move(state_)};
    }
  }

public:
  template <class Fn> [[nodiscard]] auto branch(Fn &&function) && {
    auto selected = std::forward<Fn>(function)(
        StageRef<R, stage::Bounded<Count>>{state_, values_, count_});
    using Selected = std::remove_cvref_t<decltype(selected)>;
    static_assert(detail::is_selection<Selected> ||
                      detail::is_stage_ref<Selected> ||
                      detail::is_record<Selected>,
                  "compute branch must return a stage, record, or outputs");
    if constexpr (detail::is_selection<Selected>) {
      state_.reset();
      return std::move(selected).template bind<A...>();
    } else if constexpr (detail::is_record<Selected>) {
      auto terminal = outputs(selected);
      state_.reset();
      return std::move(terminal).template bind<A...>();
    } else {
      using U = typename Selected::Value;
      using Card = typename Selected::Cardinality;
      if constexpr (detail::is_bounded_stage<Card>) {
        return Flow<U(A...), Card, Inputs>{std::move(state_), selected.value_,
                                           selected.count_};
      } else {
        detail::flow_pick(state_, selected.value_);
        return Flow<U(A...), Card, Inputs>{std::move(state_)};
      }
    }
  }
  template <class Fn>
  [[nodiscard]] auto map(const std::string_view name, Fn &&function) && {
    auto expressions = detail::make_expr();
    Expr<R> argument{
        detail::flow_expression_input<R>(state_, expressions, values_, 0u)};
    auto expression = detail::element(function, argument);
    using Expression = std::remove_cvref_t<decltype(expression)>;
    const std::array inputs{values_};
    if constexpr (detail::ComputeExpr<Expression>) {
      using U = detail::ExprValueT<Expression>;
      static_assert(detail::map_result<R, U>,
                    "compute map may change width only through mask");
      const std::uint32_t output = detail::flow_map_value_controlled(
          state_, inputs, name, expression.ref_,
          {.count = count_,
           .capacity = detail::flow_value_count(state_, values_)});
      return Flow<U(A...), stage::Bounded<Count>, Inputs>{std::move(state_),
                                                          output, count_};
    } else {
      static_assert(detail::is_expr_record<Expression>,
                    "compute bounded map must return an expression or record");
      static_assert(
          detail::ExprRecordStage<Expression,
                                  stage::Bounded<Count>>::template accepts<R>,
          "all fused record fields must match the input width");
      const auto refs = detail::ExprRecordAccess::refs(expression);
      const auto outputs = detail::flow_map_multi_controlled(
          state_, inputs, name, refs,
          {.count = count_,
           .capacity = detail::flow_value_count(state_, values_)});
      auto ids = detail::record_ids<Expression, stage::Bounded<Count>>(outputs,
                                                                       count_);
      if (outputs.size() != Expression::size) {
        detail::flow_pick(state_, 0u);
      }
      using Schema =
          detail::ExprRecordSchemaT<Expression, stage::Bounded<Count>>;
      return Flow<Schema(A...), stage::Bounded<Count>, Inputs>{
          std::move(state_), ids};
    }
  }
  template <class Fn> [[nodiscard]] auto filter(Fn &&function) && {
    auto filtered =
        StageRef<R, stage::Bounded<Count>>{state_, values_, count_}.filter(
            std::forward<Fn>(function));
    return Flow<R(A...), stage::Bounded<Count>, Inputs>{
        std::move(state_), filtered.value_, filtered.count_};
  }
  template <std::size_t N, class StepFn, class ConvergedFn>
  [[nodiscard]] auto unroll(StepFn &&step, ConvergedFn &&converged) && {
    auto remaining = StageRef<R, stage::Bounded<Count>>{state_, values_, count_}
                         .template unroll<N>(
                             std::forward<StepFn>(step),
                             std::forward<ConvergedFn>(converged));
    return Flow<R(A...), stage::Bounded<Count>, Inputs>{
        std::move(state_), remaining.value_, remaining.count_};
  }
  template <class CountFn, class EmitFn>
  [[nodiscard]] auto expand(const MaxItems bound, CountFn &&count_function,
                            EmitFn &&emit_function) &&
    requires detail::IntegerValue<R>
  {
    auto expanded =
        StageRef<R, stage::Bounded<Count>>{state_, values_, count_}.expand(
            bound, std::forward<CountFn>(count_function),
            std::forward<EmitFn>(emit_function));
    using Expanded = std::remove_cvref_t<decltype(expanded)>;
    using Card = stage::Bounded<CountFor<R>>;
    if constexpr (detail::is_record<Expanded>) {
      using Schema = detail::NodeSchemaT<Expanded>;
      return Flow<Schema(A...), Card, Inputs>{
          std::move(state_), detail::NodeAccess::ids(expanded)};
    } else {
      using U = typename Expanded::Value;
      return Flow<U(A...), Card, Inputs>{std::move(state_), expanded.value_,
                                         expanded.count_};
    }
  }
  template <class Fn>
  [[nodiscard]] auto group_by(Fn &&function) &&
    requires detail::IntegerValue<R>
  {
    auto grouped =
        StageRef<R, stage::Bounded<Count>>{state_, values_, count_}.group_by(
            std::forward<Fn>(function));
    using Key = typename decltype(grouped)::KeyType;
    return Flow<R(A...), stage::Grouped<Key, std::uint32_t>>{
        std::move(state_), std::move(grouped)};
  }
  template <class Range, class LeftKeyFn, class RightKeyFn, class EmitFn>
    requires(std::same_as<Inputs, input::Bound> && detail::IntegerValue<R> &&
             detail::BorrowedRange<Range, R>)
  [[nodiscard]] auto
  join(const MaxMatches bound, Range &right, LeftKeyFn &&left_key_function,
       RightKeyFn &&right_key_function, EmitFn &&emit_function) && {
    const std::span<const R> input{right};
    const auto side = detail::flow_side(
        state_, {input.data(), input.size(), detail::type<R>()}, true,
        detail::storage_format<R>());
    auto joined =
        StageRef<R, stage::Bounded<Count>>{state_, values_, count_}.join(
            bound, StageRef<R, stage::Exact>{state_, side},
            std::forward<LeftKeyFn>(left_key_function),
            std::forward<RightKeyFn>(right_key_function),
            std::forward<EmitFn>(emit_function));
    using Joined = std::remove_cvref_t<decltype(joined)>;
    using Card = stage::Bounded<CountFor<R>>;
    if constexpr (detail::is_record<Joined>) {
      using Schema = detail::NodeSchemaT<Joined>;
      return Flow<Schema(A..., R), Card, Inputs>{
          std::move(state_), detail::NodeAccess::ids(joined)};
    } else {
      using U = typename Joined::Value;
      return Flow<U(A..., R), Card, Inputs>{std::move(state_), joined.value_,
                                            joined.count_};
    }
  }
  template <class LeftKeyFn, class RightKeyFn, class EmitFn>
  [[nodiscard]] auto join(const MaxMatches bound, const std::size_t right_count,
                          LeftKeyFn &&left_key_function,
                          RightKeyFn &&right_key_function,
                          EmitFn &&emit_function) &&
    requires(std::same_as<Inputs, input::Deferred> && detail::IntegerValue<R>)
  {
    const std::uint32_t side =
        detail::flow_side(state_, {nullptr, right_count, detail::type<R>()},
                          false, detail::storage_format<R>());
    auto joined =
        StageRef<R, stage::Bounded<Count>>{state_, values_, count_}.join(
            bound, StageRef<R, stage::Exact>{state_, side},
            std::forward<LeftKeyFn>(left_key_function),
            std::forward<RightKeyFn>(right_key_function),
            std::forward<EmitFn>(emit_function));
    using Joined = std::remove_cvref_t<decltype(joined)>;
    using Card = stage::Bounded<CountFor<R>>;
    if constexpr (detail::is_record<Joined>) {
      using Schema = detail::NodeSchemaT<Joined>;
      return Flow<Schema(A..., R), Card, Inputs>{
          std::move(state_), detail::NodeAccess::ids(joined)};
    } else {
      using U = typename Joined::Value;
      return Flow<U(A..., R), Card, Inputs>{std::move(state_), joined.value_,
                                            joined.count_};
    }
  }
  [[nodiscard]] auto count() && {
    detail::flow_pick(state_, count_);
    return Flow<Count(A...), stage::Scalar, Inputs>{std::move(state_)};
  }
  [[nodiscard]] Flow &&scan(const Scan operation) && {
    detail::flow_pick(state_, values_);
    detail::flow_bounded_scan(state_, count_, operation);
    values_ = detail::flow_value(state_);
    return std::move(*this);
  }
  [[nodiscard]] Flow<R(A...), stage::Scalar, Inputs>
  reduce(const Reduce operation = Reduce::Sum) && {
    detail::flow_bounded_reduce(state_, count_, operation);
    return Flow<R(A...), stage::Scalar, Inputs>{std::move(state_)};
  }
  [[nodiscard]] Flow &&sort() && {
    detail::flow_bounded_sort(state_, count_, false);
    values_ = detail::flow_value(state_);
    return std::move(*this);
  }
  [[nodiscard]] Flow<std::uint32_t(A...), stage::Bounded<Count>, Inputs>
  argsort() && {
    detail::flow_bounded_sort(state_, count_, true);
    values_ = detail::flow_value(state_);
    return {std::move(state_), values_, count_};
  }
  [[nodiscard]] Flow &&window(const WindowSpec options = {}) && {
    auto next =
        StageRef<R, stage::Bounded<Count>>{state_, values_, count_}.window(
            options);
    values_ = next.value_;
    return std::move(*this);
  }
  [[nodiscard]] Result<std::vector<R>> collect() &&
    requires std::same_as<Inputs, input::Bound>
  {
    const std::array<std::uint32_t, 2u> selected{values_, count_};
    detail::flow_outputs(state_, selected);
    auto recipe = std::move(state_);
    auto compiled = detail::compile_flow(recipe);
    if (!compiled) {
      return Result<std::vector<R>>::fail(compiled.reason());
    }
    auto outputs = detail::run_host_outputs<Bounded<R, Count>>(
        std::move(compiled).value(), detail::flow_bindings(recipe));
    if (!outputs) {
      return Result<std::vector<R>>::fail(outputs.reason());
    }
    return Result<std::vector<R>>::success(std::move(std::get<0>(*outputs)));
  }
  [[nodiscard]] auto compile_async() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<Bounded<R, Count>(A...)>> compile() &&
    requires std::same_as<Inputs, input::Deferred>
  {
    const std::array<std::uint32_t, 2u> selected{values_, count_};
    detail::flow_outputs(state_, selected);
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<Bounded<R, Count>(A...)>>::fail(compiled.reason());
    }
    return Result<Program<Bounded<R, Count>(A...)>>::success(
        Program<Bounded<R, Count>(A...)>{std::move(compiled).value()});
  }

private:
  friend class FlowBuilder;
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state, const std::uint32_t values,
       const std::uint32_t count)
      : state_(std::move(state)), values_(values), count_(count) {}
  std::shared_ptr<detail::FlowState> state_;
  std::uint32_t values_{};
  std::uint32_t count_{};
};

template <class... Schema, class Count, class Inputs, class... A>
  requires detail::InputMode<Inputs>
class Flow<Record<Schema...>(A...), stage::Bounded<Count>, Inputs> final {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;

  template <class Fn> [[nodiscard]] auto branch(Fn &&function) && {
    auto selected =
        std::forward<Fn>(function)(RecordRef<Schema...>{state_, values_});
    using Selected = std::remove_cvref_t<decltype(selected)>;
    static_assert(
        detail::is_selection<Selected> || detail::is_record<Selected> ||
            detail::is_stage_ref<Selected>,
        "compute record branch must return a stage, record, or outputs");
    if constexpr (detail::is_selection<Selected>) {
      state_.reset();
      return std::move(selected).template bind<A...>();
    } else if constexpr (detail::is_record<Selected>) {
      auto terminal = outputs(selected);
      state_.reset();
      return std::move(terminal).template bind<A...>();
    } else {
      using U = typename Selected::Value;
      using Card = typename Selected::Cardinality;
      if constexpr (detail::is_bounded_stage<Card>) {
        return Flow<U(A...), Card, Inputs>{std::move(state_), selected.value_,
                                           selected.count_};
      } else {
        detail::flow_pick(state_, selected.value_);
        return Flow<U(A...), Card, Inputs>{std::move(state_)};
      }
    }
  }

  [[nodiscard]] auto compile_async() && {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<Outputs<Schema...>(A...)>> compile() && {
    detail::flow_outputs(state_, values_);
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<Outputs<Schema...>(A...)>>::fail(compiled.reason());
    }
    return Result<Program<Outputs<Schema...>(A...)>>::success(
        Program<Outputs<Schema...>(A...)>{std::move(compiled).value()});
  }

  [[nodiscard]] Result<std::tuple<detail::HostValueT<Schema>...>> collect() &&
    requires std::same_as<Inputs, input::Bound>
  {
    detail::flow_outputs(state_, values_);
    auto recipe = std::move(state_);
    auto compiled = detail::compile_flow(recipe);
    if (!compiled) {
      return Result<std::tuple<detail::HostValueT<Schema>...>>::fail(
          compiled.reason());
    }
    return detail::run_host_outputs<Schema...>(std::move(compiled).value(),
                                               detail::flow_bindings(recipe));
  }

private:
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state,
       const std::array<std::uint32_t,
                        detail::schema_leaf_count<Record<Schema...>>>
           values)
      : state_(std::move(state)), values_(values) {}
  std::shared_ptr<detail::FlowState> state_;
  std::array<std::uint32_t, detail::schema_leaf_count<Record<Schema...>>>
      values_{};
};
} // namespace rund::compute
