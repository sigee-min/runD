#ifndef RUND_COMPUTE_FLOW_STAGE_MEMBERS
#include <rund/compute/flow/stage.hpp>
#else
public:
  using Value = T;
  using Cardinality = Card;

  template <class Fn> [[nodiscard]] decltype(auto) pipe(Fn &&function) const {
    return std::forward<Fn>(function)(*this);
  }
  template <class RhsCard>
  [[nodiscard]] auto solve(const StageRef<T, RhsCard> &rhs,
                           const FactorOp operation,
                           const std::size_t rhs_cols = 1u) const
    requires(detail::FixedValue<T> && detail::is_matrix_solve_stage<Card> &&
             detail::matrix_rhs_stage_compatible<Card, RhsCard>)
  {
    using Matrix = detail::MatrixStageTraits<Card>;
    if (state_ != rhs.state_) {
      detail::flow_reject(state_, Reason::SolveShapeMismatch);
    }
    if constexpr (detail::is_matrix_stage<RhsCard>) {
      if (rhs.shape_.rows != shape_.rows || rhs.shape_.cols != rhs_cols ||
          rhs.shape_.batches != shape_.batches) {
        detail::flow_reject(state_, Reason::SolveShapeMismatch);
      }
    }
    const detail::SolveIds result = detail::flow_matrix_solve(
        state_, operation, value_, rhs.value_, shape_.rows, shape_.cols,
        rhs_cols, shape_.batches);
    return StageRef<
        T, stage::Solve<Matrix::rows, stage::Dynamic, Matrix::batches>>{
        state_, result.values, result.status,
        MatrixShape{shape_.rows, rhs_cols, shape_.batches}};
  }
  template <FactorOp Op, std::size_t RhsCols, class RhsCard>
  [[nodiscard]] auto solve(const StageRef<T, RhsCard> &rhs) const
    requires(detail::FixedValue<T> && detail::is_matrix_solve_stage<Card> &&
             (Op == FactorOp::Lu || Op == FactorOp::Qr ||
              Op == FactorOp::Cholesky) &&
             RhsCols != 0u &&
             detail::matrix_rhs_static_compatible<Card, RhsCard, RhsCols>)
  {
    using Matrix = detail::MatrixStageTraits<Card>;
    if (state_ != rhs.state_) {
      detail::flow_reject(state_, Reason::SolveShapeMismatch);
    }
    if constexpr (detail::is_matrix_stage<RhsCard>) {
      if (rhs.shape_.rows != shape_.rows || rhs.shape_.cols != RhsCols ||
          rhs.shape_.batches != shape_.batches) {
        detail::flow_reject(state_, Reason::SolveShapeMismatch);
      }
    }
    const detail::SolveIds result =
        detail::flow_matrix_solve(state_, Op, value_, rhs.value_, shape_.rows,
                                  shape_.cols, RhsCols, shape_.batches);
    return StageRef<T, stage::Solve<Matrix::rows, RhsCols, Matrix::batches>>{
        state_, result.values, result.status,
        MatrixShape{shape_.rows, RhsCols, shape_.batches}};
  }
  [[nodiscard]] auto values() const
    requires detail::is_solve_stage<Card>
  {
    using Solve = detail::SolveStageTraits<Card>;
    return StageRef<T, stage::Matrix<Solve::rows, Solve::cols, Solve::batches>>{
        state_, value_, 0u, shape_};
  }
  [[nodiscard]] StageRef<std::uint32_t, stage::Exact> status() const
    requires detail::is_solve_stage<Card>
  {
    return {state_, count_};
  }
  template <class Fn>
  [[nodiscard]] auto map(const std::string_view name, Fn &&function) const
    requires detail::is_element_stage<Card>
  {
    auto expressions = detail::make_expr();
    Expr<T> argument{
        detail::flow_expression_input<T>(state_, expressions, value_, 0u)};
    auto expression = detail::element(function, argument);
    const std::array inputs{value_};
    using Expression = std::remove_cvref_t<decltype(expression)>;
    if constexpr (detail::ComputeExpr<Expression>) {
      using U = detail::ExprValueT<Expression>;
      static_assert(detail::map_result<T, U>,
                    "compute map may change width only through mask");
      const auto control = detail::stage_control<Card>(state_, value_, count_);
      return StageRef<U, Card>{state_,
                               detail::flow_map_value_controlled(
                                   state_, inputs, name, expression.ref_,
                                   control),
                               count_};
    } else {
      static_assert(detail::is_expr_record<Expression>,
                    "compute map must return an expression or record");
      static_assert(
          detail::ExprRecordStage<Expression, Card>::template accepts<T>,
          "all fused record fields must match the input width");
      const auto refs = detail::ExprRecordAccess::refs(expression);
      const auto outputs = detail::flow_map_multi_controlled(
          state_, inputs, name, refs,
          detail::stage_control<Card>(state_, value_, count_));
      auto ids = detail::record_ids<Expression, Card>(outputs, count_);
      if (outputs.size() != Expression::size) {
        detail::flow_pick(state_, 0u);
      }
      using Result = detail::ExprRecordStageT<Expression, Card>;
      return Result{state_, ids};
    }
  }
  template <class Fn>
  [[nodiscard]] auto combine(const std::string_view name,
                             const StageRef<T, Card> &other,
                             Fn &&function) const
    requires detail::is_element_stage<Card>
  {
    auto expressions = detail::make_expr();
    Expr<T> first{
        detail::flow_expression_input<T>(state_, expressions, value_, 0u)};
    Expr<T> second{detail::flow_expression_input<T>(state_, expressions,
                                                    other.value_, 1u)};
    auto expression = detail::element(function, first, second);
    static_assert(detail::ComputeExpr<decltype(expression)>,
                  "compute branch combine must return a compute expression");
    using U = detail::ExprValueT<decltype(expression)>;
    static_assert(detail::map_result<T, U>,
                  "compute map may change width only through mask");
    if (state_ != other.state_) {
      detail::flow_pick(state_, 0u);
    }
    if constexpr (detail::is_bounded_stage<Card>) {
      if (count_ != other.count_) {
        detail::flow_pick(state_, 0u);
      }
    }
    const std::array inputs{value_, other.value_};
    return StageRef<U, Card>{state_,
                             detail::flow_map_value_controlled(
                                 state_, inputs, name, expression.ref_,
                                 detail::stage_control<Card>(state_, value_,
                                                             count_)),
                             count_};
  }
  template <class Fn>
  [[nodiscard]] auto combine(const std::string_view name,
                             const StageRef<T, stage::Scalar> &scalar,
                             Fn &&function) const
    requires(std::same_as<Card, stage::Exact> || detail::is_bounded_stage<Card>)
  {
    if (state_ != scalar.state_) {
      detail::flow_pick(state_, 0u);
    }
    const std::size_t count = detail::flow_value_count(state_, value_);
    const std::uint32_t zeros = detail::flow_zero(state_, count);
    const std::array gather_inputs{scalar.value_, zeros};
    const std::uint32_t broadcast =
        detail::flow_binary_values(state_, detail::Primitive::Gather,
                                   gather_inputs, detail::type<T>(), count, {});
    auto expressions = detail::make_expr();
    Expr<T> first{
        detail::flow_expression_input<T>(state_, expressions, value_, 0u)};
    Expr<T> second{
        detail::flow_expression_input<T>(state_, expressions, broadcast, 1u)};
    auto expression = detail::element(function, first, second);
    static_assert(detail::ComputeExpr<decltype(expression)>,
                  "compute scalar combine must return a compute expression");
    using U = detail::ExprValueT<decltype(expression)>;
    static_assert(detail::map_result<T, U>,
                  "compute map may change width only through mask");
    const std::array inputs{value_, broadcast};
    return StageRef<U, Card>{state_,
                             detail::flow_map_value_controlled(
                                 state_, inputs, name, expression.ref_,
                                 detail::stage_control<Card>(state_, value_,
                                                             count_)),
                             count_};
  }
#endif
