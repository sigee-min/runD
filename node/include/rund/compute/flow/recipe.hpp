#pragma once

#include <rund/compute/flow/branch.hpp>

namespace rund::compute {
template <class... Inputs>
class Flow<detail::InputSet<Inputs...>, stage::Exact, input::Deferred> final {
  static_assert(sizeof...(Inputs) > 0u,
                "compute input flow requires at least one input");

public:
  Flow(const Flow &) = delete;
  Flow &operator=(const Flow &) = delete;
  Flow(Flow &&) noexcept = default;
  Flow &operator=(Flow &&) noexcept = default;

  template <detail::ComputeValue T>
  [[nodiscard]] auto zip_input(const std::size_t count) && {
    const std::uint32_t value = detail::flow_independent_input(
        state_, {nullptr, count, detail::type<T>()},
        detail::storage_format<T>());
    std::array<std::uint32_t, sizeof...(Inputs) + 1u> values{};
    std::copy(values_.begin(), values_.end(), values.begin());
    values.back() = value;
    return Flow<detail::InputSet<Inputs..., T>, stage::Exact, input::Deferred>{
        std::move(state_), values};
  }

  template <class Fn> [[nodiscard]] auto branch(Fn &&function) && {
    auto selected =
        std::apply(std::forward<Fn>(function),
                   make_stages(std::index_sequence_for<Inputs...>{}));
    using Selected = std::remove_cvref_t<decltype(selected)>;
    static_assert(
        detail::is_selection<Selected> || detail::is_stage_ref<Selected> ||
            detail::is_record<Selected>,
        "compute input branch must return a stage, record, or outputs");
    if constexpr (detail::is_selection<Selected>) {
      state_.reset();
      return std::move(selected).template bind<Inputs...>();
    } else if constexpr (detail::is_record<Selected>) {
      auto terminal = outputs(selected);
      state_.reset();
      return std::move(terminal).template bind<Inputs...>();
    } else {
      using U = typename Selected::Value;
      using Card = typename Selected::Cardinality;
      if constexpr (detail::is_bounded_stage<Card>) {
        return detail::DeferredFlow<U(Inputs...), Card>{
            std::move(state_), selected.value_, selected.count_};
      } else {
        detail::flow_pick(state_, selected.value_);
        return detail::DeferredFlow<U(Inputs...), Card>{std::move(state_)};
      }
    }
  }

  template <class Fn>
  [[nodiscard]] auto map(const std::string_view name, Fn &&function) && {
    const std::size_t count = detail::flow_value_count(state_, values_.front());
    for (const std::uint32_t value : values_) {
      if (detail::flow_value_count(state_, value) != count) {
        detail::flow_reject(state_, Reason::ZipShapeMismatch);
        break;
      }
    }
    auto expressions = detail::make_expr();
    auto arguments =
        make_arguments(expressions, std::index_sequence_for<Inputs...>{});
    auto expression = std::apply(
        [&](auto &...argument) {
          return detail::element(function, argument...);
        },
        arguments);
    using Expression = std::remove_cvref_t<decltype(expression)>;
    if constexpr (detail::ComputeExpr<Expression>) {
      using U = detail::ExprValueT<Expression>;
      static_assert(sizeof(U) == sizeof(First),
                    "compute input map output must preserve scalar width");
      const std::array refs{expression.ref_};
      const auto outputs = detail::flow_map_multi(state_, values_, name, refs);
      detail::flow_pick(state_, outputs.size() == 1u ? outputs.front() : 0u);
      return detail::DeferredFlow<U(Inputs...), stage::Exact>{
          std::move(state_)};
    } else {
      static_assert(detail::is_expr_record<Expression>,
                    "compute input map must return an expression or record");
      static_assert(
          detail::ExprRecordStage<Expression,
                                  stage::Exact>::template accepts<First>,
          "all fused record fields must preserve the input scalar width");
      const auto refs = detail::ExprRecordAccess::refs(expression);
      const auto outputs = detail::flow_map_multi(state_, values_, name, refs);
      std::array<std::uint32_t, Expression::size> ids{};
      if (outputs.size() == ids.size()) {
        std::copy(outputs.begin(), outputs.end(), ids.begin());
      } else {
        detail::flow_pick(state_, 0u);
      }
      using Schema = detail::ExprRecordSchemaT<Expression, stage::Exact>;
      return detail::DeferredFlow<Schema(Inputs...), stage::Exact>{
          std::move(state_), ids};
    }
  }

private:
  using First = std::tuple_element_t<0u, std::tuple<Inputs...>>;
  friend class FlowBuilder;
  template <class, class, class> friend class Flow;
  Flow(std::shared_ptr<detail::FlowState> state,
       const std::array<std::uint32_t, sizeof...(Inputs)> values)
      : state_(std::move(state)), values_(values) {}

  template <std::size_t... I>
  [[nodiscard]] auto
  make_arguments(const std::shared_ptr<detail::ExprState> &expressions,
                 std::index_sequence<I...>) const {
    return std::tuple{Expr<std::tuple_element_t<I, std::tuple<Inputs...>>>{
        detail::flow_expression_input<
            std::tuple_element_t<I, std::tuple<Inputs...>>>(
            state_, expressions, values_[I],
            static_cast<std::uint32_t>(I))}...};
  }

  template <std::size_t... I>
  [[nodiscard]] auto make_stages(std::index_sequence<I...>) const {
    return std::tuple{
        StageRef<std::tuple_element_t<I, std::tuple<Inputs...>>, stage::Exact>{
            state_, values_[I]}...};
  }

  std::shared_ptr<detail::FlowState> state_;
  std::array<std::uint32_t, sizeof...(Inputs)> values_{};
};

} // namespace rund::compute
