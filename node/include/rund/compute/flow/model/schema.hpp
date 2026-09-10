#pragma once

#include <rund/compute/flow/model/expression.hpp>

namespace rund::compute::detail {

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

} // namespace rund::compute::detail
