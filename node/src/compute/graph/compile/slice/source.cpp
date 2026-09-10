#include <rund/compute/abi/ids.hpp>
#include <rund/compute/abi/primitive.hpp>
#include "internal.hpp"

#include <algorithm>
#include <span>
#include <variant>
#include <vector>

namespace rund::compute::detail::graph_compile::slice_detail {
namespace {

[[nodiscard]] bool valid_expression(const ExprRef &expression,
                                    const Type type) noexcept {
  return expression.state != nullptr && expression.state->status &&
         expression.node != 0u &&
         expression.node <= expression.state->nodes.size() &&
         expression.type == type && expression.fixed_format == FixedFormat{};
}

[[nodiscard]] bool empty(const FlowControl control) noexcept {
  return control.count == 0u && control.predicate == 0u &&
         control.capacity == 0u && control.predicate_expected == 0u &&
         control.iteration == 0u;
}

[[nodiscard]] bool plain_value(const GraphValue &value, const Type type,
                               const std::size_t count) noexcept {
  return value.type == type && value.fixed_format == FixedFormat{} &&
         value.count == count && value.active == 0u && value.parent == 0u;
}

[[nodiscard]] bool direct_expression(const ExprRef &expression,
                                     const Type type) noexcept {
  if (expression.state == nullptr || !expression.state->status ||
      expression.node == 0u ||
      expression.node > expression.state->nodes.size() ||
      expression.type != type || expression.fixed_format != FixedFormat{}) {
    return false;
  }
  return std::none_of(
      expression.state->nodes.begin(), expression.state->nodes.end(),
      [](const ExprNode node) { return node.operation == ExprOp::Index; });
}

[[nodiscard]] bool valid_page_value(const GraphState &graph,
                                    const std::uint32_t value,
                                    const Type type) noexcept {
  return value != 0u && value <= graph.values.size() &&
         plain_value(graph.values[value - 1u], type, graph.count);
}

} // namespace

Result<SliceSource>
inspect_reduce(const std::shared_ptr<ProgramState> &program) noexcept {
  if (program == nullptr || program->device == nullptr ||
      program->canonical_graph == nullptr) {
    return Result<SliceSource>::fail(Reason::ProgramInvalid);
  }
  const GraphState &graph = *program->canonical_graph;
  const Type type =
      program->input_types.empty() ? Type::I32 : program->input_types.front();
  if (!graph.status || graph.device != program->device || graph.count == 0u ||
      graph.inputs.empty() || graph.inputs.size() > MaxMapInputs ||
      graph.outputs.size() != 1u || !graph.bounded_inputs.empty() ||
      !graph.identity_outputs.empty() || graph.steps.size() < 2u ||
      program->input_types.size() != graph.inputs.size() ||
      program->input_sizes.size() != graph.inputs.size() ||
      program->input_formats.size() != graph.inputs.size() ||
      program->output_types.size() != 1u ||
      program->output_sizes.size() != 1u || type != Type::U64 ||
      !std::all_of(program->input_types.begin(), program->input_types.end(),
                   [type](const Type input) { return input == type; }) ||
      !std::all_of(
          program->input_sizes.begin(), program->input_sizes.end(),
          [&graph](const std::size_t count) { return count == graph.count; }) ||
      !std::all_of(
          program->input_formats.begin(), program->input_formats.end(),
          [](const FixedFormat format) { return format == FixedFormat{}; }) ||
      program->output_types.front() != type ||
      program->output_sizes.front() != 1u ||
      !std::all_of(graph.inputs.begin(), graph.inputs.end(),
                   [&graph, type](const std::uint32_t input) {
                     return valid_page_value(graph, input, type);
                   })) {
    return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
  }
  std::vector<bool> produced(graph.values.size() + 1u, false);
  for (const std::uint32_t input : graph.inputs) {
    produced[input] = true;
  }
  for (std::size_t index = 0u; index + 1u < graph.steps.size(); ++index) {
    const auto *const map = std::get_if<MapStep>(&graph.steps[index]);
    if (map == nullptr || !empty(map->control) || !map->reads.empty() ||
        map->expressions.empty() ||
        !std::all_of(map->expressions.begin(), map->expressions.end(),
                     [type](const ExprRef &expression) {
                       return direct_expression(expression, type);
                     }) ||
        !graph.value_ids.valid(map->inputs) ||
        !graph.value_ids.valid(map->outputs)) {
      return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
    }
    const std::span<const std::uint32_t> inputs =
        graph.value_ids.view(map->inputs);
    const std::span<const std::uint32_t> outputs =
        graph.value_ids.view(map->outputs);
    if (inputs.empty() || outputs.empty() || inputs.size() > MaxMapInputs ||
        outputs.size() > MaxOutputs ||
        outputs.size() != map->expressions.size()) {
      return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
    }
    for (const std::uint32_t input : inputs) {
      if (!valid_page_value(graph, input, type) || !produced[input]) {
        return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
      }
    }
    for (const std::uint32_t output : outputs) {
      if (!valid_page_value(graph, output, type) || produced[output]) {
        return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
      }
      produced[output] = true;
    }
  }
  const auto *const reduce = std::get_if<GraphPrimitive>(&graph.steps.back());
  const bool reduce_operation =
      reduce != nullptr && reduce->primitive == Primitive::Reduce &&
      ((!reduce->options.flag &&
        (reduce->options.mode == static_cast<std::uint32_t>(Reduce::Sum) ||
         reduce->options.mode == static_cast<std::uint32_t>(Reduce::Min) ||
         reduce->options.mode == static_cast<std::uint32_t>(Reduce::Max))) ||
       (reduce->options.flag &&
        reduce->options.mode == static_cast<std::uint32_t>(Reduce::Sum)));
  if (!reduce_operation || reduce->options.first != 0u ||
      reduce->options.second != 0u || reduce->options.third != 0u ||
      reduce->options.fourth != 0u || reduce->options.extra != 0u ||
      !empty(reduce->control) || !graph.value_ids.valid(reduce->inputs) ||
      !graph.value_ids.valid(reduce->outputs)) {
    return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
  }
  const std::span<const std::uint32_t> inputs =
      graph.value_ids.view(reduce->inputs);
  const std::span<const std::uint32_t> outputs =
      graph.value_ids.view(reduce->outputs);
  if (inputs.size() != 1u || outputs.size() != 1u ||
      !valid_page_value(graph, inputs.front(), type) ||
      !produced[inputs.front()] || outputs.front() != reduce->output ||
      outputs.front() != graph.outputs.front() || outputs.front() == 0u ||
      outputs.front() > graph.values.size() ||
      !plain_value(graph.values[outputs.front() - 1u], type, 1u)) {
    return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
  }
  return Result<SliceSource>::success(SliceSource{.graph = &graph,
                                                  .reduce = reduce,
                                                  .type = type,
                                                  .capacity = graph.count});
}

Result<SliceSource>
inspect_pointwise(const std::shared_ptr<ProgramState> &program) noexcept {
  if (program == nullptr || program->device == nullptr ||
      program->canonical_graph == nullptr) {
    return Result<SliceSource>::fail(Reason::ProgramInvalid);
  }
  const GraphState &graph = *program->canonical_graph;
  const Type type =
      program->input_types.empty() ? Type::I32 : program->input_types.front();
  if (!graph.status || graph.device != program->device || graph.count == 0u ||
      graph.inputs.empty() || graph.inputs.size() > MaxMapInputs ||
      graph.outputs.size() != 1u || !graph.bounded_inputs.empty() ||
      !graph.identity_outputs.empty() || graph.steps.size() < 2u ||
      (type != Type::U32 && type != Type::U64) ||
      program->input_types.size() != graph.inputs.size() ||
      program->input_sizes.size() != graph.inputs.size() ||
      program->input_formats.size() != graph.inputs.size() ||
      program->output_types.size() != 1u ||
      program->output_sizes.size() != 1u ||
      program->output_formats.size() != 1u ||
      !std::all_of(program->input_types.begin(), program->input_types.end(),
                   [type](const Type input) { return input == type; }) ||
      !std::all_of(
          program->input_sizes.begin(), program->input_sizes.end(),
          [&graph](const std::size_t count) { return count == graph.count; }) ||
      !std::all_of(
          program->input_formats.begin(), program->input_formats.end(),
          [](const FixedFormat format) { return format == FixedFormat{}; }) ||
      program->output_types.front() != type ||
      program->output_sizes.front() != graph.count ||
      program->output_formats.front() != FixedFormat{} ||
      !std::all_of(graph.inputs.begin(), graph.inputs.end(),
                   [&graph, type](const std::uint32_t input) {
                     return valid_page_value(graph, input, type);
                   })) {
    return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
  }
  std::vector<bool> produced(graph.values.size() + 1u, false);
  for (const std::uint32_t input : graph.inputs) {
    produced[input] = true;
  }
  for (const GraphStep &step : graph.steps) {
    const auto *const map = std::get_if<MapStep>(&step);
    if (map == nullptr || !empty(map->control) || !map->reads.empty() ||
        map->expressions.empty() ||
        !std::all_of(map->expressions.begin(), map->expressions.end(),
                     [type](const ExprRef &expression) {
                       return valid_expression(expression, type);
                     }) ||
        !graph.value_ids.valid(map->inputs) ||
        !graph.value_ids.valid(map->outputs)) {
      return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
    }
    const std::span<const std::uint32_t> inputs =
        graph.value_ids.view(map->inputs);
    const std::span<const std::uint32_t> outputs =
        graph.value_ids.view(map->outputs);
    if (inputs.empty() || outputs.empty() || inputs.size() > MaxMapInputs ||
        outputs.size() > MaxOutputs ||
        outputs.size() != map->expressions.size()) {
      return Result<SliceSource>::fail(Reason::PrimitiveUnsupported);
    }
    for (const std::uint32_t input : inputs) {
      if (!valid_page_value(graph, input, type) || !produced[input]) {
        return Result<SliceSource>::fail(Reason::GraphBindingInvalid);
      }
    }
    for (const std::uint32_t output : outputs) {
      if (!valid_page_value(graph, output, type) || produced[output]) {
        return Result<SliceSource>::fail(Reason::GraphBindingInvalid);
      }
      produced[output] = true;
    }
  }
  const std::uint32_t output = graph.outputs.front();
  if (!valid_page_value(graph, output, type) || !produced[output]) {
    return Result<SliceSource>::fail(Reason::GraphBindingInvalid);
  }
  const auto *const last = std::get_if<MapStep>(&graph.steps.back());
  if (last == nullptr || !graph.value_ids.valid(last->outputs)) {
    return Result<SliceSource>::fail(Reason::GraphBindingInvalid);
  }
  const std::span<const std::uint32_t> terminal_outputs =
      graph.value_ids.view(last->outputs);
  if (std::find(terminal_outputs.begin(), terminal_outputs.end(), output) ==
      terminal_outputs.end()) {
    return Result<SliceSource>::fail(Reason::GraphBindingInvalid);
  }
  return Result<SliceSource>::success(
      SliceSource{.graph = &graph, .type = type, .capacity = graph.count});
}

} // namespace rund::compute::detail::graph_compile::slice_detail
