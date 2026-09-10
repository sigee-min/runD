#pragma once

#include <rund/compute/flow/branch.hpp>

namespace rund::compute {
template <class R, class... A>
class Flow<R(A...), stage::Scalar, input::Bound> final
    : public detail::StagePipe<Flow<R(A...), stage::Scalar, input::Bound>> {
public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;

private:
  friend class detail::StagePipe<Flow<R(A...), stage::Scalar>>;
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
    return Flow<U(A...), stage::Scalar>{std::move(state_)};
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
    return Flow<U(A...), stage::Scalar>{std::move(state_)};
  }
  [[nodiscard]] Result<std::vector<R>> collect() && {
    auto recipe = std::move(state_);
    auto compiled = detail::compile_flow(recipe);
    if (!compiled) {
      return Result<std::vector<R>>::fail(compiled.reason());
    }
    return detail::run_host_views<R>(std::move(compiled).value(),
                                     detail::flow_bindings(recipe));
  }

private:
  template <class, class, class> friend class Flow;
  explicit Flow(std::shared_ptr<detail::FlowState> state)
      : state_(std::move(state)) {}
  std::shared_ptr<detail::FlowState> state_;
};

} // namespace rund::compute
