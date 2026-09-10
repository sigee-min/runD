#include "internal.hpp"

#include <rund/compute/abi/graph.hpp>

#include <algorithm>
#include <new>
#include <span>
#include <variant>
#include <vector>

namespace rund::compute::detail::graph_compile::slice_detail {

bool bind_count(GraphState &graph, const std::uint32_t count,
                const std::size_t capacity) {
  try {
    graph.bounded_inputs.push_back(
        BoundedInputSchema{.count = count, .capacity = capacity});
    return true;
  } catch (const std::bad_alloc &) {
    graph.status = Status::fail(Reason::GraphCapacity);
    return false;
  }
}

Result<std::shared_ptr<ProgramState>>
compile_map_prefix(const SliceSource source,
                   std::vector<std::uint32_t> &stage_inputs,
                   std::vector<std::uint32_t> &stage_outputs) {
  const auto graph =
      make_graph(source.graph->device, "tiled-map-fused", source.capacity);
  if (graph == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
  }
  std::vector<std::uint32_t> projected(source.graph->values.size() + 1u, 0u);
  std::vector<std::uint32_t> inputs;
  inputs.reserve(source.graph->inputs.size());
  stage_inputs.assign(source.graph->inputs.begin(), source.graph->inputs.end());
  for (const std::uint32_t value : source.graph->inputs) {
    const std::uint32_t input =
        graph_input_count(graph, source.type, source.capacity);
    if (input == 0u || value >= projected.size()) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          graph->status.reason());
    }
    inputs.push_back(input);
    projected[value] = input;
  }
  const std::uint32_t count = graph_input_count(graph, Type::U64, 1u);
  if (count == 0u || !bind_count(*graph, count, source.capacity)) {
    return Result<std::shared_ptr<ProgramState>>::fail(graph->status.reason());
  }
  for (std::size_t index = 0u; index + 1u < source.graph->steps.size();
       ++index) {
    const MapStep &map = std::get<MapStep>(source.graph->steps[index]);
    const std::span<const std::uint32_t> source_inputs =
        source.graph->value_ids.view(map.inputs);
    const std::span<const std::uint32_t> source_outputs =
        source.graph->value_ids.view(map.outputs);
    std::vector<std::uint32_t> mapped;
    mapped.reserve(source_inputs.size());
    for (const std::uint32_t input : source_inputs) {
      if (input >= projected.size() || projected[input] == 0u) {
        return Result<std::shared_ptr<ProgramState>>::fail(
            Reason::GraphBindingInvalid);
      }
      mapped.push_back(projected[input]);
    }
    const ValueIds outputs = graph_map_multi_controlled(
        graph, mapped, map.expressions,
        FlowControl{.count = count, .capacity = source.capacity}, map.name,
        source.capacity);
    if (!graph->status || outputs.size() != source_outputs.size()) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          graph->status ? Reason::GraphBindingInvalid : graph->status.reason());
    }
    for (std::size_t output = 0u; output < source_outputs.size(); ++output) {
      if (source_outputs[output] >= projected.size() ||
          projected[source_outputs[output]] != 0u) {
        return Result<std::shared_ptr<ProgramState>>::fail(
            Reason::GraphBindingInvalid);
      }
      projected[source_outputs[output]] = outputs[output];
    }
  }
  const std::span<const std::uint32_t> terminal_inputs =
      source.graph->value_ids.view(source.reduce->inputs);
  std::vector<std::uint32_t> outputs;
  outputs.reserve(terminal_inputs.size());
  stage_outputs.assign(terminal_inputs.begin(), terminal_inputs.end());
  for (const std::uint32_t output : terminal_inputs) {
    if (output >= projected.size() || projected[output] == 0u) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          Reason::GraphBindingInvalid);
    }
    outputs.push_back(projected[output]);
  }
  graph_outputs(graph, outputs);
  if (!graph->status) {
    return Result<std::shared_ptr<ProgramState>>::fail(graph->status.reason());
  }
  std::vector<Type> input_types(inputs.size(), source.type);
  input_types.push_back(Type::U64);
  std::vector<Type> output_types(outputs.size(), source.type);
  return compile_graph(graph, input_types, output_types);
}

bool requires_external_wavefront(const SliceSource source) noexcept {
  if (source.graph == nullptr || source.graph->steps.size() < 3u) {
    return false;
  }
  for (std::size_t index = 1u; index + 1u < source.graph->steps.size();
       ++index) {
    const auto *const map = std::get_if<MapStep>(&source.graph->steps[index]);
    if (map == nullptr || !source.graph->value_ids.valid(map->inputs)) {
      return false;
    }
    const std::span<const std::uint32_t> inputs =
        source.graph->value_ids.view(map->inputs);
    if (std::any_of(inputs.begin(), inputs.end(),
                    [source](const std::uint32_t input) {
                      return std::find(source.graph->inputs.begin(),
                                       source.graph->inputs.end(),
                                       input) != source.graph->inputs.end();
                    })) {
      return true;
    }
  }
  return false;
}

Result<std::shared_ptr<ProgramState>>
compile_map_stage(const SliceSource source, const std::size_t stage,
                  std::vector<std::uint32_t> &stage_inputs,
                  std::vector<std::uint32_t> &stage_outputs) {
  if (source.graph == nullptr || stage >= source.graph->steps.size()) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphInvalid);
  }
  const auto *const map = std::get_if<MapStep>(&source.graph->steps[stage]);
  if (map == nullptr || !source.graph->value_ids.valid(map->inputs) ||
      !source.graph->value_ids.valid(map->outputs)) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphInvalid);
  }
  const std::span<const std::uint32_t> source_inputs =
      source.graph->value_ids.view(map->inputs);
  const std::span<const std::uint32_t> source_outputs =
      source.graph->value_ids.view(map->outputs);
  if (source_inputs.empty() || source_outputs.empty() ||
      source_outputs.size() != map->expressions.size()) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        Reason::GraphBindingInvalid);
  }

  const auto graph =
      make_graph(source.graph->device, "tiled-map-stage", source.capacity);
  if (graph == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
  }
  std::vector<std::uint32_t> inputs;
  inputs.reserve(source_inputs.size());
  for (std::size_t index = 0u; index < source_inputs.size(); ++index) {
    const std::uint32_t input =
        graph_input_count(graph, source.type, source.capacity);
    if (input == 0u) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          graph->status.reason());
    }
    inputs.push_back(input);
  }
  const std::uint32_t count = graph_input_count(graph, Type::U64, 1u);
  if (count == 0u || !bind_count(*graph, count, source.capacity)) {
    return Result<std::shared_ptr<ProgramState>>::fail(graph->status.reason());
  }
  const ValueIds outputs = graph_map_multi_controlled(
      graph, inputs, map->expressions,
      FlowControl{.count = count, .capacity = source.capacity}, map->name,
      source.capacity);
  if (!graph->status || outputs.size() != source_outputs.size()) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        graph->status ? Reason::GraphBindingInvalid : graph->status.reason());
  }
  graph_outputs(graph, outputs);
  if (!graph->status) {
    return Result<std::shared_ptr<ProgramState>>::fail(graph->status.reason());
  }
  stage_inputs.assign(source_inputs.begin(), source_inputs.end());
  stage_outputs.assign(source_outputs.begin(), source_outputs.end());
  std::vector<Type> input_types(inputs.size(), source.type);
  input_types.push_back(Type::U64);
  std::vector<Type> output_types(outputs.size(), source.type);
  return compile_graph(graph, input_types, output_types);
}

} // namespace rund::compute::detail::graph_compile::slice_detail
