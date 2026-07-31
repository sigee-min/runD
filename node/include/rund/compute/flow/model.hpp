#pragma once
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <rund/compute/abi/flow.hpp>
#include <rund/compute/cache.hpp>
#include <rund/compute/device.hpp>
#include <rund/compute/expr/recipe.hpp>
#include <rund/compute/expr/select.hpp>
#include <rund/compute/ops.hpp>
#include <rund/compute/program.hpp>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
namespace rund::compute {
enum class FactorOp : unsigned char;
enum class SpectrumOp : unsigned char;
enum class SpectrumVectors : unsigned char;
namespace stage {
inline constexpr std::size_t Dynamic = static_cast<std::size_t>(-1);
struct Exact final {};
template <class Count> struct Bounded final {};
template <class Key, class Count> struct Grouped final {};
struct Complex final {};
template <std::size_t Rows = Dynamic, std::size_t Cols = Dynamic,
          std::size_t Batches = Dynamic>
struct Matrix final {};
struct Scalar final {};
template <FactorOp Op, std::size_t Rows = Dynamic, std::size_t Cols = Dynamic,
          std::size_t Batches = Dynamic>
struct Factor final {};
template <std::size_t Rows = Dynamic, std::size_t Cols = Dynamic,
          std::size_t Batches = Dynamic>
struct Solve final {};
template <SpectrumOp Op, SpectrumVectors V, std::size_t Rows = Dynamic,
          std::size_t Cols = Dynamic, std::size_t Batches = Dynamic>
struct Spectrum final {};
} // namespace stage
namespace input {
// Flow type identity only; neither marker is stored in a Flow instance.
struct Bound final {};
struct Deferred final {};
} // namespace input
namespace detail {
template <class Mode>
concept InputMode =
    std::same_as<Mode, input::Bound> || std::same_as<Mode, input::Deferred>;
template <class... Values> struct InputSet final {};
template <class Tag, class Node> struct FieldNode final {
  Node value;
};
} // namespace detail
template <class Signature, class Stage = stage::Exact,
          class Mode = input::Bound>
class Flow;
namespace detail {
template <class Signature, class Stage = stage::Exact>
using DeferredFlow = Flow<Signature, Stage, input::Deferred>;
template <class Recipe>
  requires requires(Recipe &&recipe) { std::move(recipe).compile(); }
[[nodiscard]] auto compile_async(const std::shared_ptr<FlowState> &state,
                                 Recipe recipe);
} // namespace detail
template <class T, class Card = stage::Exact> class StageRef;
template <class... Stages> class ZipRef;
template <class Key, class Value, class Count = std::uint32_t> class Groups;
template <class T, class Count = CountFor<T>> class GroupValuesRef;
template <class... Schema> class RecordRef;
template <class... Schema> class Selection;
namespace detail {
struct NodeAccess;
template <class Schema> struct SchemaRef;
struct StageRefAccess final {
  template <class T, class Card>
  [[nodiscard]] static StageRef<T, Card>
  make(std::shared_ptr<FlowState> state, const std::uint32_t value,
       const std::uint32_t count = 0u) {
    return StageRef<T, Card>{std::move(state), value, count};
  }
  template <class T, class Card>
  [[nodiscard]] static const std::shared_ptr<FlowState> &
  state(const StageRef<T, Card> &value) noexcept {
    return value.state_;
  }
  template <class T, class Card>
  [[nodiscard]] static std::uint32_t
  id(const StageRef<T, Card> &value) noexcept {
    return value.value_;
  }
  template <class T, class Card>
  [[nodiscard]] static std::uint32_t
  count(const StageRef<T, Card> &value) noexcept {
    return value.count_;
  }
};
template <class T> inline constexpr bool is_bounded_stage = false;
template <class Count>
inline constexpr bool is_bounded_stage<stage::Bounded<Count>> = true;
struct NoSide final {};
template <class T, class Card> struct ResidentCount final {
  using Type = CountFor<T>;
};
template <class T, class Count>
struct ResidentCount<T, stage::Bounded<Count>> final {
  using Type = Count;
};
template <class T, class Card>
using ResidentCountT = typename ResidentCount<T, Card>::Type;
template <class T> inline constexpr bool is_matrix_stage = false;
template <std::size_t Rows, std::size_t Cols, std::size_t Batches>
inline constexpr bool is_matrix_stage<stage::Matrix<Rows, Cols, Batches>> =
    true;
template <class T> inline constexpr bool is_solve_stage = false;
template <std::size_t Rows, std::size_t Cols, std::size_t Batches>
inline constexpr bool is_solve_stage<stage::Solve<Rows, Cols, Batches>> = true;
template <class T> struct MatrixStageTraits;
template <std::size_t Rows, std::size_t Cols, std::size_t Batches>
struct MatrixStageTraits<stage::Matrix<Rows, Cols, Batches>> final {
  inline static constexpr std::size_t rows = Rows;
  inline static constexpr std::size_t cols = Cols;
  inline static constexpr std::size_t batches = Batches;
};
template <class T> struct SolveStageTraits;
template <std::size_t Rows, std::size_t Cols, std::size_t Batches>
struct SolveStageTraits<stage::Solve<Rows, Cols, Batches>> final {
  inline static constexpr std::size_t rows = Rows;
  inline static constexpr std::size_t cols = Cols;
  inline static constexpr std::size_t batches = Batches;
};
template <class T>
inline constexpr bool is_element_stage =
    std::same_as<T, stage::Exact> || std::same_as<T, stage::Scalar> ||
    is_bounded_stage<T>;
template <class T>
inline constexpr bool is_matrix_solve_stage = [] {
  if constexpr (!is_matrix_stage<T>) {
    return false;
  } else {
    using Matrix = MatrixStageTraits<T>;
    return Matrix::rows == stage::Dynamic || Matrix::cols == stage::Dynamic ||
           Matrix::rows == Matrix::cols;
  }
}();
template <class MatrixCard, class RhsCard>
inline constexpr bool matrix_rhs_stage_compatible = [] {
  if constexpr (!is_matrix_stage<MatrixCard>) {
    return false;
  } else if constexpr (std::same_as<RhsCard, stage::Exact>) {
    return true;
  } else if constexpr (is_matrix_stage<RhsCard>) {
    using Left = MatrixStageTraits<MatrixCard>;
    using Right = MatrixStageTraits<RhsCard>;
    return (Left::rows == stage::Dynamic || Right::rows == stage::Dynamic ||
            Left::rows == Right::rows) &&
           (Left::batches == stage::Dynamic ||
            Right::batches == stage::Dynamic ||
            Left::batches == Right::batches);
  } else {
    return false;
  }
}();
template <class MatrixCard, class RhsCard, std::size_t RhsCols>
inline constexpr bool matrix_rhs_static_compatible = [] {
  if constexpr (!matrix_rhs_stage_compatible<MatrixCard, RhsCard>) {
    return false;
  } else if constexpr (std::same_as<RhsCard, stage::Exact>) {
    return true;
  } else {
    using Right = MatrixStageTraits<RhsCard>;
    return Right::cols == stage::Dynamic || Right::cols == RhsCols;
  }
}();
template <class T>
concept IntegerValue =
    std::same_as<T, std::int32_t> || std::same_as<T, std::uint32_t> ||
    std::same_as<T, std::int64_t> || std::same_as<T, std::uint64_t>;
template <class Card>
[[nodiscard]] inline FlowControl
stage_control(const std::shared_ptr<FlowState> &state,
              const std::uint32_t value, const std::uint32_t count) {
  if constexpr (is_bounded_stage<Card>) {
    return FlowControl{.count = count,
                       .capacity = flow_value_count(state, value)};
  } else {
    return {};
  }
}
template <class T> inline constexpr bool is_initializer_list = false;
template <class T>
inline constexpr bool is_initializer_list<std::initializer_list<T>> = true;
template <class Range, class = void> struct BorrowedRangeTraits final {};
template <class Range>
struct BorrowedRangeTraits<
    Range, std::void_t<decltype(std::span{std::declval<Range &>()})>>
    final {
  using Span = decltype(std::span{std::declval<Range &>()});
  using Value = std::remove_cv_t<typename Span::element_type>;
};
template <class Range, class T>
concept BorrowedRange =
    ComputeValue<std::remove_cv_t<T>> &&
    (!is_initializer_list<std::remove_cv_t<Range>>) &&
    requires { typename BorrowedRangeTraits<Range>::Value; } &&
    std::same_as<typename BorrowedRangeTraits<Range>::Value,
                 std::remove_cv_t<T>> &&
    requires(Range &range) { std::span<const std::remove_cv_t<T>>{range}; };
template <class Range>
concept ComputeRange = requires {
  typename BorrowedRangeTraits<Range>::Value;
} && BorrowedRange<Range, typename BorrowedRangeTraits<Range>::Value>;
template <class Range>
using BorrowedRangeValue = typename BorrowedRangeTraits<Range>::Value;
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
template <class T, class Card> struct StageSchema;
template <class T> struct StageSchema<T, stage::Exact> final {
  using Type = T;
};
template <class T> struct StageSchema<T, stage::Scalar> final {
  using Type = Scalar<T>;
};
template <class T, class Count>
struct StageSchema<T, stage::Bounded<Count>> final {
  using Type = Bounded<T, Count>;
};
template <class T, class Card>
using StageSchemaT = typename StageSchema<T, Card>::Type;
template <class T> struct NodeSchema final {
  static constexpr bool valid = false;
};
template <class T, class Card> struct NodeSchema<StageRef<T, Card>> final {
  static constexpr bool valid = true;
  using Type = StageSchemaT<T, Card>;
};
template <class... Schema> struct NodeSchema<RecordRef<Schema...>> final {
  static constexpr bool valid = true;
  using Type = Record<Schema...>;
};
template <class Tag, class Node> struct NodeSchema<FieldNode<Tag, Node>> final {
  static constexpr bool valid = NodeSchema<Node>::valid;
  using Type = Field<Tag, typename NodeSchema<Node>::Type>;
};
template <class T>
concept SelectionNode = NodeSchema<std::remove_cvref_t<T>>::valid;
template <class T>
using NodeSchemaT = typename NodeSchema<std::remove_cvref_t<T>>::Type;
template <std::size_t I, class... Schema>
inline constexpr std::size_t schema_offset = [] {
  static_assert(I < sizeof...(Schema), "compute record field out of range");
  constexpr std::array sizes{schema_leaf_count<Schema>...};
  std::size_t offset = 0u;
  for (std::size_t index = 0u; index < I; ++index) {
    offset += sizes[index];
  }
  return offset;
}();
template <class Schema> struct SchemaTag final {
  using Type = void;
};
template <class Tag, class Schema> struct SchemaTag<Field<Tag, Schema>> final {
  using Type = Tag;
};
template <class Tag, class... Schema>
inline constexpr std::size_t field_index = [] {
  constexpr std::array matches{
      std::is_same_v<Tag, typename SchemaTag<Schema>::Type>...};
  std::size_t found = sizeof...(Schema);
  std::size_t count = 0u;
  for (std::size_t index = 0u; index < matches.size(); ++index) {
    if (matches[index]) {
      found = index;
      ++count;
    }
  }
  if (count != 1u) {
    return sizeof...(Schema);
  }
  return found;
}();
template <class Schema> struct SchemaRef final {
  template <std::size_t N>
  [[nodiscard]] static auto make(const std::shared_ptr<FlowState> &state,
                                 const std::array<std::uint32_t, N> &values,
                                 const std::size_t offset) {
    return StageRef<Schema, stage::Exact>{state, values[offset]};
  }
};
template <class T> struct SchemaRef<Scalar<T>> final {
  template <std::size_t N>
  [[nodiscard]] static auto make(const std::shared_ptr<FlowState> &state,
                                 const std::array<std::uint32_t, N> &values,
                                 const std::size_t offset) {
    return StageRef<T, stage::Scalar>{state, values[offset]};
  }
};
template <class T, class Count> struct SchemaRef<Bounded<T, Count>> final {
  template <std::size_t N>
  [[nodiscard]] static auto make(const std::shared_ptr<FlowState> &state,
                                 const std::array<std::uint32_t, N> &values,
                                 const std::size_t offset) {
    return StageRef<T, stage::Bounded<Count>>{state, values[offset],
                                              values[offset + 1u]};
  }
};
template <class... Schema> struct SchemaRef<Record<Schema...>> final {
  template <std::size_t N>
  [[nodiscard]] static auto make(const std::shared_ptr<FlowState> &state,
                                 const std::array<std::uint32_t, N> &values,
                                 const std::size_t offset) {
    std::array<std::uint32_t, schema_leaf_count<Record<Schema...>>> fields{};
    for (std::size_t index = 0u; index < fields.size(); ++index) {
      fields[index] = values[offset + index];
    }
    return RecordRef<Schema...>{state, fields};
  }
};
template <class Tag, class Schema> struct SchemaRef<Field<Tag, Schema>> final {
  template <std::size_t N>
  [[nodiscard]] static auto make(const std::shared_ptr<FlowState> &state,
                                 const std::array<std::uint32_t, N> &values,
                                 const std::size_t offset) {
    return SchemaRef<Schema>::make(state, values, offset);
  }
};
template <class T> inline constexpr bool is_stage_result = false;
template <class Signature, class Stage, class Mode>
inline constexpr bool is_stage_result<Flow<Signature, Stage, Mode>> = true;
template <class T> inline constexpr bool is_selection = false;
template <class... T>
inline constexpr bool is_selection<Selection<T...>> = true;
template <class T> inline constexpr bool is_record = false;
template <class... T> inline constexpr bool is_record<RecordRef<T...>> = true;
template <class T> inline constexpr bool is_stage_ref = false;
template <class T, class Card>
inline constexpr bool is_stage_ref<StageRef<T, Card>> = true;
template <class Record, class Card> struct ExprRecordStage;
template <class Field, class Card> struct ExprFieldStage;
template <class Tag, class T, class Card>
struct ExprFieldStage<ExprField<Tag, T>, Card> final {
  using Schema = Field<Tag, StageSchemaT<T, Card>>;
  template <class Input>
  static constexpr bool accepts = sizeof(T) == sizeof(Input);
};
template <class Tag, class... Fields, class Card>
struct ExprFieldStage<ExprRecordField<Tag, Fields...>, Card> final {
  using Schema =
      Field<Tag, Record<typename ExprFieldStage<Fields, Card>::Schema...>>;
  template <class Input>
  static constexpr bool accepts =
      (ExprFieldStage<Fields, Card>::template accepts<Input> && ...);
};
template <class... Fields, class Card>
struct ExprRecordStage<ExprRecord<Fields...>, Card> final {
  using Schema = Record<typename ExprFieldStage<Fields, Card>::Schema...>;
  using Type = RecordRef<typename ExprFieldStage<Fields, Card>::Schema...>;
  template <class Input>
  static constexpr bool accepts =
      (ExprFieldStage<Fields, Card>::template accepts<Input> && ...);
};
template <class Record, class Card>
using ExprRecordStageT = typename ExprRecordStage<Record, Card>::Type;
template <class Record, class Card>
using ExprRecordSchemaT = typename ExprRecordStage<Record, Card>::Schema;
template <class Record, class Card>
[[nodiscard]] auto record_ids(const ValueIds &outputs,
                              const std::uint32_t count) {
  using Schema = ExprRecordSchemaT<Record, Card>;
  std::array<std::uint32_t, schema_leaf_count<Schema>> ids{};
  std::size_t offset = 0u;
  for (const std::uint32_t output : outputs) {
    ids[offset++] = output;
    if constexpr (is_bounded_stage<Card>) {
      ids[offset++] = count;
    }
  }
  return ids;
}
template <class Expression, class Count>
[[nodiscard]] auto bounded_emit(const std::shared_ptr<FlowState> &state,
                                const std::span<const std::uint32_t> inputs,
                                const std::string_view name,
                                const Expression &expression,
                                const std::uint32_t count) {
  using Value = std::remove_cvref_t<Expression>;
  using Card = stage::Bounded<Count>;
  if constexpr (ComputeExpr<Value>) {
    using U = ExprValueT<Value>;
    return StageRef<U, Card>{
        state, flow_map_value(state, inputs, name, ExprAccess::ref(expression)),
        count};
  } else {
    static_assert(is_expr_record<Value>,
                  "compute bounded emit must return an expression or record");
    const auto refs = ExprRecordAccess::refs(expression);
    const auto outputs = flow_map_multi(state, inputs, name, refs);
    auto ids = record_ids<Value, Card>(outputs, count);
    if (outputs.size() != Value::size) {
      flow_pick(state, 0u);
    }
    using Result = ExprRecordStageT<Value, Card>;
    return Result{state, ids};
  }
}
template <class Expression, class Count>
[[nodiscard]] auto
bounded_emit_reject(const std::shared_ptr<FlowState> &state) {
  using Value = std::remove_cvref_t<Expression>;
  using Card = stage::Bounded<Count>;
  if constexpr (ComputeExpr<Value>) {
    return StageRef<ExprValueT<Value>, Card>{state, 0u, 0u};
  } else {
    static_assert(is_expr_record<Value>,
                  "compute bounded emit must return an expression or record");
    using Schema = ExprRecordSchemaT<Value, Card>;
    return ExprRecordStageT<Value, Card>{
        state, std::array<std::uint32_t, schema_leaf_count<Schema>>{}};
  }
}
template <ComputeValue T>
[[nodiscard]] ExprRef
flow_expression_input(const std::shared_ptr<FlowState> &flow,
                      const std::shared_ptr<ExprState> &expressions,
                      const std::uint32_t value, const std::uint32_t index) {
  return input(expressions, type<T>(), index, flow_value_format(flow, value));
}
template <class Stage> class StagePipe {
public:
  template <class Fn>
    requires requires(Stage &&stage, Fn &&function) {
      static_cast<Stage &&>(stage).pipe_stage(std::forward<Fn>(function));
    }
  [[nodiscard]] decltype(auto) pipe(Fn &&function) && {
    return static_cast<Stage &&>(*this).pipe_stage(std::forward<Fn>(function));
  }

protected:
  StagePipe() = default;
};
struct FlowFactory final {
  template <ComputeValue T>
  [[nodiscard]] static Flow<T(T), stage::Exact> make(Target target,
                                                     std::span<const T> input);
};
} // namespace detail

template <class Fn, detail::ComputeValue... Values>
[[nodiscard]] auto capture(Fn &&function, Values... values) {
  using Function = std::remove_cvref_t<Fn>;
  static_assert(std::is_empty_v<Function>,
                "compute::capture requires a captureless element function");
  static_assert(std::is_trivially_copy_constructible_v<Function>,
                "compute::capture requires a trivial element function");
  return detail::CapturedElement<Function, Values...>{
      std::forward<Fn>(function), std::tuple<Values...>{std::move(values)...}};
}

} // namespace rund::compute
