#pragma once

#include <rund/compute/flow/stage.hpp>

namespace rund::compute {
template <class... Stages> class ZipRef final {
  static_assert(sizeof...(Stages) >= 2u,
                "compute zip requires at least two stages");
  using StageTuple = std::tuple<Stages...>;
  using First = std::tuple_element_t<0u, StageTuple>;

public:
  using Cardinality = typename First::Cardinality;
  explicit ZipRef(Stages... stages) : stages_(std::move(stages)...) {}
  ZipRef(const ZipRef &) = default;
  ZipRef(ZipRef &&) noexcept = default;
  ZipRef &operator=(const ZipRef &) = default;
  ZipRef &operator=(ZipRef &&) noexcept = default;

  template <class Fn>
  [[nodiscard]] auto map(const std::string_view name, Fn &&function) const {
    const auto &first = std::get<0u>(stages_);
    const auto state = detail::StageRefAccess::state(first);
    const std::size_t expected =
        detail::flow_value_count(state, detail::StageRefAccess::id(first));
    const std::uint32_t lineage = detail::StageRefAccess::count(first);
    std::apply(
        [&](const auto &...stage) {
          (([&] {
             if (detail::StageRefAccess::state(stage) != state ||
                 detail::flow_value_count(
                     state, detail::StageRefAccess::id(stage)) != expected) {
               detail::flow_pick(state, 0u);
             }
             if constexpr (detail::is_bounded_stage<Cardinality>) {
               if (detail::StageRefAccess::count(stage) != lineage) {
                 detail::flow_pick(state, 0u);
               }
             }
           }()),
           ...);
        },
        stages_);

    auto expressions = detail::make_expr();
    auto arguments =
        make_arguments(expressions, std::index_sequence_for<Stages...>{});
    auto expression = std::apply(
        [&](auto &...argument) {
          return detail::element(function, argument...);
        },
        arguments);
    using Expression = std::remove_cvref_t<decltype(expression)>;
    const auto inputs = input_ids(std::index_sequence_for<Stages...>{});
    if constexpr (detail::ComputeExpr<Expression>) {
      using U = detail::ExprValueT<Expression>;
      static_assert(sizeof(U) == sizeof(typename First::Value),
                    "compute zip map output must preserve scalar width");
      const std::uint32_t output = [&] {
        if constexpr (detail::is_bounded_stage<Cardinality>) {
          return detail::flow_map_value_controlled(
              state, inputs, name, expression.ref_,
              {.count = lineage, .capacity = expected});
        } else {
          return detail::flow_map_value(state, inputs, name, expression.ref_);
        }
      }();
      return StageRef<U, Cardinality>{state, output, lineage};
    } else {
      static_assert(detail::is_expr_record<Expression>,
                    "compute zip map must return an expression or record");
      static_assert(
          detail::ExprRecordStage<Expression, Cardinality>::template accepts<
              typename First::Value>,
          "all fused record fields must preserve the zip scalar width");
      const auto refs = detail::ExprRecordAccess::refs(expression);
      const auto outputs = [&] {
        if constexpr (detail::is_bounded_stage<Cardinality>) {
          return detail::flow_map_multi_controlled(
              state, inputs, name, refs,
              {.count = lineage, .capacity = expected});
        } else {
          return detail::flow_map_multi(state, inputs, name, refs);
        }
      }();
      auto ids = detail::record_ids<Expression, Cardinality>(outputs, lineage);
      if (outputs.size() != Expression::size) {
        detail::flow_pick(state, 0u);
      }
      using Result = detail::ExprRecordStageT<Expression, Cardinality>;
      return Result{state, ids};
    }
  }

private:
  template <std::size_t... I>
  [[nodiscard]] auto
  make_arguments(const std::shared_ptr<detail::ExprState> &expressions,
                 std::index_sequence<I...>) const {
    return std::tuple{Expr<typename std::tuple_element_t<I, StageTuple>::Value>{
        detail::flow_expression_input<
            typename std::tuple_element_t<I, StageTuple>::Value>(
            detail::StageRefAccess::state(std::get<I>(stages_)), expressions,
            detail::StageRefAccess::id(std::get<I>(stages_)),
            static_cast<std::uint32_t>(I))}...};
  }
  template <std::size_t... I>
  [[nodiscard]] auto input_ids(std::index_sequence<I...>) const {
    return std::array<std::uint32_t, sizeof...(I)>{
        detail::StageRefAccess::id(std::get<I>(stages_))...};
  }
  StageTuple stages_;
};

namespace detail {
template <class First, class... Rest>
concept CompatibleZip =
    sizeof...(Rest) > 0u && is_stage_ref<std::remove_cvref_t<First>> &&
    (is_stage_ref<std::remove_cvref_t<Rest>> && ...) &&
    (std::same_as<typename std::remove_cvref_t<First>::Cardinality,
                  typename std::remove_cvref_t<Rest>::Cardinality> &&
     ...) &&
    (std::same_as<typename std::remove_cvref_t<First>::Cardinality,
                  stage::Exact> ||
     is_bounded_stage<typename std::remove_cvref_t<First>::Cardinality>) &&
    ((sizeof(typename std::remove_cvref_t<First>::Value) ==
      sizeof(typename std::remove_cvref_t<Rest>::Value)) &&
     ...);
} // namespace detail

template <class First, class... Rest>
  requires detail::CompatibleZip<First, Rest...>
[[nodiscard]] auto zip(const First &first, const Rest &...rest) {
  return ZipRef<First, Rest...>{first, rest...};
}

} // namespace rund::compute
