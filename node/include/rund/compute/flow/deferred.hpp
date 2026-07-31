#pragma once

#include <rund/compute/flow/branch.hpp>

namespace rund::compute {
template <class R, class... A>
class Flow<R(A...), stage::Exact, input::Deferred> final
    : public detail::StagePipe<detail::DeferredFlow<R(A...), stage::Exact>> {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;

private:
  friend class detail::StagePipe<detail::DeferredFlow<R(A...), stage::Exact>>;
  template <class Fn> [[nodiscard]] auto pipe_stage(Fn &&function) && {
    auto next = std::forward<Fn>(function)(
        StageRef<R, stage::Exact>{state_, detail::flow_value(state_)});
    using Next = std::remove_cvref_t<decltype(next)>;
    static_assert(detail::is_stage_ref<Next>,
                  "compute pipe must return a typed stage");
    if (state_ != detail::StageRefAccess::state(next)) {
      detail::flow_pick(state_, 0u);
    }
    using U = typename Next::Value;
    using Card = typename Next::Cardinality;
    if constexpr (detail::is_bounded_stage<Card>) {
      return detail::DeferredFlow<U(A...), Card>{std::move(state_), next.value_,
                                                 next.count_};
    } else {
      detail::flow_pick(state_, next.value_);
      return detail::DeferredFlow<U(A...), Card>{std::move(state_)};
    }
  }

public:
  template <class Fn> [[nodiscard]] auto branch(Fn &&function) && {
    auto selected = std::forward<Fn>(function)(
        StageRef<R, stage::Exact>{state_, detail::flow_value(state_)});
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
        return detail::DeferredFlow<U(A...), Card>{
            std::move(state_), selected.value_, selected.count_};
      } else {
        detail::flow_pick(state_, selected.value_);
        return detail::DeferredFlow<U(A...), Card>{std::move(state_)};
      }
    }
  }
  template <class Fn>
  [[nodiscard]] auto map(const std::string_view name, Fn &&function) && {
    auto expressions = detail::make_expr();
    Expr<R> argument{detail::flow_expression_input<R>(
        state_, expressions, detail::flow_value(state_), 0u)};
    auto expression = detail::element(function, argument);
    using Expression = std::remove_cvref_t<decltype(expression)>;
    if constexpr (detail::ComputeExpr<Expression>) {
      using U = detail::ExprValueT<Expression>;
      static_assert(detail::map_result<R, U>,
                    "compute map may change width only through mask");
      detail::flow_map(state_, name, expression.ref_);
      return detail::DeferredFlow<U(A...), stage::Exact>{std::move(state_)};
    } else {
      static_assert(detail::is_expr_record<Expression>,
                    "compute plan map must return an expression or record");
      static_assert(detail::ExprRecordStage<Expression,
                                            stage::Exact>::template accepts<R>,
                    "all fused record fields must match the input width");
      const auto refs = detail::ExprRecordAccess::refs(expression);
      const std::array inputs{detail::flow_value(state_)};
      const auto outputs = detail::flow_map_multi(state_, inputs, name, refs);
      std::array<std::uint32_t, Expression::size> ids{};
      if (outputs.size() == ids.size()) {
        std::copy(outputs.begin(), outputs.end(), ids.begin());
      } else {
        detail::flow_pick(state_, 0u);
      }
      using Schema = detail::ExprRecordSchemaT<Expression, stage::Exact>;
      return detail::DeferredFlow<Schema(A...), stage::Exact>{std::move(state_),
                                                              ids};
    }
  }
  template <class Fn>
  [[nodiscard]] detail::DeferredFlow<R(A...), stage::Bounded<CountFor<R>>>
  filter(Fn &&function) && {
    auto expressions = detail::make_expr();
    Expr<R> argument{detail::flow_expression_input<R>(
        state_, expressions, detail::flow_value(state_), 0u)};
    auto predicate = detail::element(function, argument);
    static_assert(
        std::is_same_v<std::remove_cvref_t<decltype(predicate)>, Predicate<R>>,
        "compute filter must return a predicate of its value type");
    using Count = CountFor<R>;
    const Expr<Count> selected = detail::count_mask<Count>(predicate);
    const Expr<Count> rejected = detail::count_mask<Count>(!predicate);
    const detail::BoundedIds result =
        detail::flow_filter(state_, selected.ref_, rejected.ref_);
    return detail::DeferredFlow<R(A...), stage::Bounded<CountFor<R>>>{
        std::move(state_), result.values, result.count};
  }
  template <class CountFn, class EmitFn>
  [[nodiscard]] auto expand(const MaxItems bound, CountFn &&count_function,
                            EmitFn &&emit_function) &&
    requires detail::IntegerValue<R>
  {
    auto expanded =
        StageRef<R, stage::Exact>{state_, detail::flow_value(state_)}.expand(
            bound, std::forward<CountFn>(count_function),
            std::forward<EmitFn>(emit_function));
    using Expanded = std::remove_cvref_t<decltype(expanded)>;
    using Card = stage::Bounded<CountFor<R>>;
    if constexpr (detail::is_record<Expanded>) {
      using Schema = detail::NodeSchemaT<Expanded>;
      return detail::DeferredFlow<Schema(A...), Card>{
          std::move(state_), detail::NodeAccess::ids(expanded)};
    } else {
      using U = typename Expanded::Value;
      return detail::DeferredFlow<U(A...), Card>{
          std::move(state_), expanded.value_, expanded.count_};
    }
  }
  template <class Fn>
  [[nodiscard]] auto combine(const std::string_view name,
                             const std::size_t right_count, Fn &&function) && {
    if (right_count != detail::flow_output_count(state_)) {
      detail::flow_reject(state_, Reason::CombineShapeMismatch);
    }
    const std::uint32_t side =
        detail::flow_side(state_, {nullptr, right_count, detail::type<R>()},
                          false, detail::storage_format<R>());
    auto combined =
        StageRef<R, stage::Exact>{state_, detail::flow_value(state_)}.combine(
            name, StageRef<R, stage::Exact>{state_, side},
            std::forward<Fn>(function));
    using U = typename decltype(combined)::Value;
    detail::flow_pick(state_, combined.value_);
    return detail::DeferredFlow<U(A..., R), stage::Exact>{std::move(state_)};
  }
  template <class Fn>
  [[nodiscard]] auto group_by(Fn &&function) &&
    requires detail::IntegerValue<R>
  {
    auto grouped =
        StageRef<R, stage::Exact>{state_, detail::flow_value(state_)}.group_by(
            std::forward<Fn>(function));
    using Key = typename decltype(grouped)::KeyType;
    return Flow<R(A...), stage::Grouped<Key, std::uint32_t>>{
        std::move(state_), std::move(grouped)};
  }
  template <class LeftKeyFn, class RightKeyFn, class EmitFn>
  [[nodiscard]] auto join(const MaxMatches bound, const std::size_t right_count,
                          LeftKeyFn &&left_key_function,
                          RightKeyFn &&right_key_function,
                          EmitFn &&emit_function) &&
    requires detail::IntegerValue<R>
  {
    const std::uint32_t side =
        detail::flow_side(state_, {nullptr, right_count, detail::type<R>()},
                          false, detail::storage_format<R>());
    auto joined =
        StageRef<R, stage::Exact>{state_, detail::flow_value(state_)}.join(
            bound, StageRef<R, stage::Exact>{state_, side},
            std::forward<LeftKeyFn>(left_key_function),
            std::forward<RightKeyFn>(right_key_function),
            std::forward<EmitFn>(emit_function));
    using Joined = std::remove_cvref_t<decltype(joined)>;
    using Card = stage::Bounded<CountFor<R>>;
    if constexpr (detail::is_record<Joined>) {
      using Schema = detail::NodeSchemaT<Joined>;
      return detail::DeferredFlow<Schema(A..., R), Card>{
          std::move(state_), detail::NodeAccess::ids(joined)};
    } else {
      using U = typename Joined::Value;
      return detail::DeferredFlow<U(A..., R), Card>{
          std::move(state_), joined.value_, joined.count_};
    }
  }
  [[nodiscard]] Flow &&scan(const Scan operation) && {
    detail::flow_scan(state_, operation);
    return std::move(*this);
  }
  [[nodiscard]] detail::DeferredFlow<R(A...), stage::Scalar>
  reduce(const Reduce operation = Reduce::Sum) && {
    if (detail::flow_output_count(state_) == 0u && operation != Reduce::Sum) {
      detail::flow_reject(state_, Reason::ReduceCountZero);
    }
    detail::flow_unary(state_, detail::Primitive::Reduce, detail::type<R>(), 1u,
                       {.mode = static_cast<std::uint32_t>(operation)});
    return detail::DeferredFlow<R(A...), stage::Scalar>{std::move(state_)};
  }
  [[nodiscard]] auto count() && {
    using Count =
        std::conditional_t<sizeof(R) == 8u, std::uint64_t, std::uint32_t>;
    detail::flow_unary(state_, detail::Primitive::Reduce, detail::type<Count>(),
                       1u, {.flag = true});
    return detail::DeferredFlow<Count(A...), stage::Scalar>{std::move(state_)};
  }
  [[nodiscard]] Flow &&sort() && {
    detail::flow_unary(state_, detail::Primitive::Sort, detail::type<R>(),
                       detail::flow_output_count(state_), {});
    return std::move(*this);
  }
  [[nodiscard]] detail::DeferredFlow<std::uint32_t(A...)> argsort() && {
    detail::flow_unary(state_, detail::Primitive::Argsort,
                       detail::type<std::uint32_t>(),
                       detail::flow_output_count(state_), {});
    return detail::DeferredFlow<std::uint32_t(A...)>{std::move(state_)};
  }
  [[nodiscard]] detail::DeferredFlow<std::uint32_t(A...),
                                     stage::Bounded<std::uint32_t>>
  compact(Compact options = {}) &&
    requires std::same_as<R, std::uint32_t>
  {
    if (options.capacity == 0u) {
      options.capacity = detail::flow_output_count(state_);
    }
    const detail::BoundedIds result = detail::flow_compact_value(
        state_, detail::flow_value(state_), options.capacity);
    return {std::move(state_), result.values, result.count};
  }
  [[nodiscard]] Flow &&histogram(const Histogram options) &&
    requires std::same_as<R, std::uint32_t>
  {
    detail::flow_unary(state_, detail::Primitive::Histogram, detail::type<R>(),
                       options.bins, {.first = options.bins});
    return std::move(*this);
  }
  [[nodiscard]] Flow &&window(const WindowSpec options = {}) && {
    auto next =
        StageRef<R, stage::Exact>{state_, detail::flow_value(state_)}.window(
            options);
    detail::flow_pick(state_, next.value_);
    return std::move(*this);
  }
  [[nodiscard]] detail::DeferredFlow<R(A...), stage::Matrix<>>
  matrix(const MatrixShape shape) && {
    detail::flow_matrix_view(state_, shape.rows, shape.cols, shape.batches);
    return detail::DeferredFlow<R(A...), stage::Matrix<>>{std::move(state_),
                                                          shape};
  }
  template <std::size_t Rows, std::size_t Cols, std::size_t Batches = 1u>
    requires(Rows != 0u && Cols != 0u && Batches != 0u &&
             Rows <= static_cast<std::size_t>(-1) / Cols &&
             Rows * Cols <= static_cast<std::size_t>(-1) / Batches)
  [[nodiscard]] detail::DeferredFlow<R(A...),
                                     stage::Matrix<Rows, Cols, Batches>>
  matrix() && {
    const MatrixShape shape{Rows, Cols, Batches};
    detail::flow_matrix_view(state_, Rows, Cols, Batches);
    return detail::DeferredFlow<R(A...), stage::Matrix<Rows, Cols, Batches>>{
        std::move(state_), shape};
  }

  [[nodiscard]] detail::DeferredFlow<R(A..., R), stage::Complex> complex() &&
    requires detail::FixedValue<R>
  {
    const std::uint32_t real = detail::flow_value(state_);
    const std::uint32_t imag = detail::flow_complex_side(
        state_, {nullptr, detail::flow_output_count(state_), detail::type<R>()},
        false, detail::storage_format<R>());
    return detail::DeferredFlow<R(A..., R), stage::Complex>{std::move(state_),
                                                            real, imag};
  }
  [[nodiscard]] detail::DeferredFlow<R(A..., std::uint32_t)>
  gather(const std::size_t count) && {
    const std::uint32_t side = detail::flow_side(
        state_, {nullptr, count, detail::Type::U32}, false, {});
    detail::flow_binary(state_, detail::Primitive::Gather, side, false,
                        detail::type<R>(), count, {});
    return detail::DeferredFlow<R(A..., std::uint32_t)>{std::move(state_)};
  }
  [[nodiscard]] detail::DeferredFlow<R(A..., std::uint32_t)>
  scatter(const std::size_t index_count, Scatter options = {}) && {
    if (options.count == 0u) {
      options.count = detail::flow_output_count(state_);
    }
    const std::uint32_t side = detail::flow_side(
        state_, {nullptr, index_count, detail::Type::U32}, false, {});
    detail::flow_binary(state_, detail::Primitive::Scatter, side, false,
                        detail::type<R>(), options.count,
                        {.first = options.count});
    return detail::DeferredFlow<R(A..., std::uint32_t)>{std::move(state_)};
  }
  [[nodiscard]] detail::DeferredFlow<R(A..., std::uint32_t)>
  partition(const std::size_t count) && {
    const std::uint32_t side = detail::flow_side(
        state_, {nullptr, count, detail::Type::U32}, false, {});
    detail::flow_binary(state_, detail::Primitive::Partition, side, true,
                        detail::type<R>(), detail::flow_output_count(state_),
                        {});
    return detail::DeferredFlow<R(A..., std::uint32_t)>{std::move(state_)};
  }
  [[nodiscard]] detail::DeferredFlow<R(A..., std::uint32_t)>
  segmented_scan(const std::size_t count, const Scan operation) && {
    const std::uint32_t side = detail::flow_side(
        state_, {nullptr, count, detail::Type::U32}, false, {});
    detail::flow_binary(state_, detail::Primitive::SegmentedScan, side, false,
                        detail::type<R>(), detail::flow_output_count(state_),
                        {.mode = static_cast<std::uint32_t>(operation)});
    return detail::DeferredFlow<R(A..., std::uint32_t)>{std::move(state_)};
  }
  [[nodiscard]] detail::DeferredFlow<R(A..., std::uint32_t)>
  segmented_reduce(const std::size_t count,
                   const Reduce operation = Reduce::Sum) && {
    const std::uint32_t side = detail::flow_side(
        state_, {nullptr, count, detail::Type::U32}, false, {});
    detail::flow_binary(state_, detail::Primitive::SegmentedReduce, side, false,
                        detail::type<R>(), detail::flow_output_count(state_),
                        {.mode = static_cast<std::uint32_t>(operation)});
    return detail::DeferredFlow<R(A..., std::uint32_t)>{std::move(state_)};
  }
  [[nodiscard]] auto compile_async() && {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<R(A...)>> compile() && {
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<R(A...)>>::fail(compiled.reason());
    }
    return Result<Program<R(A...)>>::success(
        Program<R(A...)>{std::move(compiled).value()});
  }

private:
  friend class FlowBuilder;
  template <class, class, class> friend class Flow;
  explicit Flow(std::shared_ptr<detail::FlowState> state)
      : state_(std::move(state)) {}
  std::shared_ptr<detail::FlowState> state_;
};
template <class R, class... A>
class Flow<R(A...), stage::Scalar, input::Deferred> final
    : public detail::StagePipe<detail::DeferredFlow<R(A...), stage::Scalar>> {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;

private:
  friend class detail::StagePipe<detail::DeferredFlow<R(A...), stage::Scalar>>;
  template <class Fn> [[nodiscard]] auto pipe_stage(Fn &&function) && {
    auto next = std::forward<Fn>(function)(
        StageRef<R, stage::Scalar>{state_, detail::flow_value(state_)});
    using Next = std::remove_cvref_t<decltype(next)>;
    static_assert(detail::is_stage_ref<Next>,
                  "compute pipe must return a typed stage");
    static_assert(std::same_as<typename Next::Cardinality, stage::Scalar>,
                  "compute scalar pipe must preserve scalar cardinality");
    using U = typename Next::Value;
    if (state_ != detail::StageRefAccess::state(next)) {
      detail::flow_pick(state_, 0u);
    }
    detail::flow_pick(state_, next.value_);
    return detail::DeferredFlow<U(A...), stage::Scalar>{std::move(state_)};
  }

public:
  template <class Fn>
  [[nodiscard]] auto map(const std::string_view name, Fn &&function) && {
    auto expressions = detail::make_expr();
    Expr<R> argument{detail::flow_expression_input<R>(
        state_, expressions, detail::flow_value(state_), 0u)};
    auto expression = detail::element(function, argument);
    static_assert(detail::ComputeExpr<decltype(expression)>,
                  "compute scalar map must return a compute expression");
    using U = detail::ExprValueT<decltype(expression)>;
    static_assert(detail::map_result<R, U>,
                  "compute map may change width only through mask");
    detail::flow_map(state_, name, expression.ref_);
    return detail::DeferredFlow<U(A...), stage::Scalar>{std::move(state_)};
  }
  [[nodiscard]] auto compile_async() && {
    auto state = state_;
    return detail::compile_async(state, std::move(*this));
  }

  [[nodiscard]] Result<Program<R(A...)>> compile() && {
    auto compiled = detail::compile_flow(state_);
    state_.reset();
    if (!compiled) {
      return Result<Program<R(A...)>>::fail(compiled.reason());
    }
    return Result<Program<R(A...)>>::success(
        Program<R(A...)>{std::move(compiled).value()});
  }

private:
  template <class, class, class> friend class Flow;
  explicit Flow(std::shared_ptr<detail::FlowState> state)
      : state_(std::move(state)) {}
  std::shared_ptr<detail::FlowState> state_;
};

} // namespace rund::compute
