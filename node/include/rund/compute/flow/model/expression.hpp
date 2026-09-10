#pragma once

#include <rund/compute/flow/model/traits.hpp>

namespace rund::compute::detail {

template <class Fn, ComputeValue... Values> struct CapturedElement final {
  Fn function;
  std::tuple<Values...> values;
};
template <class Fn> struct NegatedElement final : Fn {
  template <class... Args>
  [[nodiscard]] constexpr auto operator()(Args &&...args) const {
    return !static_cast<const Fn &>(*this)(std::forward<Args>(args)...);
  }
};
template <class T> inline constexpr bool captured_element_function = false;
template <class Fn, ComputeValue... Values>
inline constexpr bool
    captured_element_function<CapturedElement<Fn, Values...>> = true;
template <class Fn>
inline constexpr bool captured_element_function<NegatedElement<Fn>> =
    captured_element_function<Fn>;
template <class T> struct CapturedTraits;
template <class T> using PureArgT = StaticArgT<T, 0u>;
template <class Fn, ComputeValue... Values>
struct CapturedTraits<CapturedElement<Fn, Values...>> final {
  template <class... Args, std::size_t... I, std::size_t... C>
  [[nodiscard]] static consteval auto recipe(std::index_sequence<I...>,
                                             std::index_sequence<C...>) {
    return Fn{}(StaticArgT<Args, I>{{}}...,
                StaticExpr<Values, StaticCapture<C>>{{}}...);
  }
  template <std::size_t I>
  [[nodiscard]] static const auto &
  value(const CapturedElement<Fn, Values...> &function) {
    return std::get<I>(function.values);
  }
};
template <class Fn> struct CapturedTraits<NegatedElement<Fn>> final {
  template <class... Args, std::size_t... I, std::size_t... C>
  [[nodiscard]] static consteval auto
  recipe(std::index_sequence<I...> inputs, std::index_sequence<C...> captures) {
    return !CapturedTraits<Fn>::template recipe<Args...>(inputs, captures);
  }
  template <std::size_t I>
  [[nodiscard]] static const auto &value(const NegatedElement<Fn> &function) {
    return CapturedTraits<Fn>::template value<I>(
        static_cast<const Fn &>(function));
  }
};
template <class Fn> [[nodiscard]] auto negate_element(Fn &&function) {
  return NegatedElement<std::remove_cvref_t<Fn>>{{std::forward<Fn>(function)}};
}
template <class Function, class... Args>
[[nodiscard]] consteval auto static_recipe() {
  if constexpr (captured_element_function<Function>) {
    return CapturedTraits<Function>::template recipe<Args...>(
        std::index_sequence_for<Args...>{},
        std::make_index_sequence<
            std::tuple_size_v<decltype(std::declval<Function>().values)>>{});
  } else {
    return []<std::size_t... I>(std::index_sequence<I...>) {
      return Function{}(StaticArgT<Args, I>{{}}...);
    }(std::index_sequence_for<Args...>{});
  }
}
template <std::size_t I, class T, class Function>
[[nodiscard]] ExprRef static_capture(Function &function,
                                     const std::shared_ptr<ExprState> &state) {
  const T value = CapturedTraits<Function>::template value<I>(function);
  if constexpr (FixedValue<T>) {
    return constant(state, type<T>(), static_bits(value), fixed_format<T>());
  } else {
    return constant(state, type<T>(), static_bits(value));
  }
}
template <class Fn, class... Args>
[[nodiscard]] decltype(auto) element(Fn &function, Args &&...args) {
  using Function = std::remove_cvref_t<Fn>;
  static_assert(std::is_trivially_copy_constructible_v<Function>,
                "compute element function may capture only canonical values");
  static_assert(std::is_empty_v<Function> ||
                    captured_element_function<Function>,
                "compute element function must be captureless or use "
                "compute::capture with canonical values");
  constexpr auto recipe = static_recipe<Function, Args...>();
  auto inputs = std::forward_as_tuple(args...);
  auto expression = materialize_static(recipe, function, inputs);
  if constexpr (ComputeExpr<decltype(expression)>) {
    return expression;
  } else {
    return expression;
  }
}
template <class Input, class Output>
inline constexpr bool map_result = sizeof(Input) == sizeof(Output) ||
                                   (sizeof(Input) == sizeof(std::uint64_t) &&
                                    std::same_as<Output, std::uint32_t>);
template <std::unsigned_integral Count, class T>
[[nodiscard]] Expr<Count> count_mask(const Predicate<T> &predicate) {
  return ExprAccess::make<Count>(
      make_mask(ExprAccess::ref(predicate), type<Count>()));
}
template <std::unsigned_integral Count, class T, class Node>
[[nodiscard]] constexpr auto
count_mask(const StaticPredicate<T, Node> &predicate) {
  return StaticExpr<Count, StaticUnary<ExprOp::Mask, StaticPredicate<T, Node>>>{
      {predicate}};
}

} // namespace rund::compute::detail
