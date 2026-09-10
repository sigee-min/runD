#include "semantic.hpp"

#include "../../state.hpp"

#include <rund/compute/abi/graph.hpp>

#include <algorithm>
#include <array>
#include <new>
#include <variant>
#include <vector>

namespace rund::compute::detail::graph_compile {
namespace {

[[nodiscard]] bool plain(const GraphState &graph, const std::uint32_t value,
                         const Type type, const std::size_t count) noexcept {
  return value != 0u && value <= graph.values.size() &&
         graph.values[value - 1u].type == type &&
         graph.values[value - 1u].fixed_format == FixedFormat{} &&
         graph.values[value - 1u].count == count;
}

[[nodiscard]] const GraphState *
exact_map_dag(const ProgramState &program) noexcept {
  const GraphState *const graph = program.canonical_graph.get();
  const Type type =
      program.input_types.empty() ? Type::I32 : program.input_types.front();
  const bool input_schema =
      !program.input_types.empty() &&
      program.input_types.size() == program.input_sizes.size() &&
      program.input_types.size() == program.input_formats.size() &&
      program.input_types.size() <= MaxMapInputs &&
      std::all_of(program.input_types.begin(), program.input_types.end(),
                  [type](const Type input) { return input == type; }) &&
      std::all_of(program.input_sizes.begin(), program.input_sizes.end(),
                  [&program](const std::size_t count) {
                    return count != 0u && count == program.input_sizes.front();
                  }) &&
      std::all_of(
          program.input_formats.begin(), program.input_formats.end(),
          [](const FixedFormat format) { return format == FixedFormat{}; });
  const bool graph_inputs =
      graph != nullptr && graph->inputs.size() == program.input_types.size() &&
      std::all_of(graph->inputs.begin(), graph->inputs.end(),
                  [graph, type](const std::uint32_t input) {
                    return plain(*graph, input, type, graph->count);
                  });
  return graph != nullptr && graph->status && graph->device == program.device &&
                 !graph->steps.empty() && graph_inputs &&
                 graph->outputs.size() == 1u && input_schema &&
                 program.output_types.size() == 1u &&
                 program.output_sizes.size() == 1u &&
                 program.output_formats.size() == 1u &&
                 (type == Type::U32 || type == Type::U64) &&
                 program.output_types.front() == type &&
                 program.output_formats.front() == FixedFormat{} &&
                 program.input_sizes.front() == program.output_sizes.front() &&
                 graph->count == program.input_sizes.front() &&
                 plain(*graph, graph->outputs.front(), type, graph->count) &&
                 std::all_of(graph->steps.begin(), graph->steps.end(),
                             [](const GraphStep &step) {
                               return std::holds_alternative<MapStep>(step);
                             })
             ? graph
             : nullptr;
}

} // namespace

Result<std::shared_ptr<ProgramState>>
compile_service_free_map_program(const std::shared_ptr<ProgramState> &program) {
  const GraphState *const source =
      program == nullptr ? nullptr : exact_map_dag(*program);
  if (source == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        Reason::PrimitiveUnsupported);
  }
  auto semantic = compile_service_free_map_semantic_inputs(
      *source, program->input_types.front(), source->count, source->inputs,
      source->outputs.front());
  if (!semantic || semantic.value() == nullptr ||
      semantic.value()->canonical_graph == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        semantic ? Reason::ProgramInvalid : semantic.reason());
  }
  const GraphState &composed = *semantic.value()->canonical_graph;
  const auto *const map = composed.steps.size() == 1u
                              ? std::get_if<MapStep>(&composed.steps.front())
                              : nullptr;
  if (map == nullptr || map->expressions.size() != 1u ||
      !composed.value_ids.valid(map->inputs) ||
      composed.value_ids.view(map->inputs).size() != source->inputs.size()) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::ProgramInvalid);
  }

  const auto graph =
      make_graph(program->device, "service-free-map-dag", source->count);
  if (graph == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
  }
  std::vector<std::uint32_t> inputs;
  try {
    inputs.reserve(source->inputs.size());
    for (std::size_t index = 0u; index < source->inputs.size(); ++index) {
      const std::uint32_t input =
          graph_input_count(graph, program->input_types[index], source->count);
      if (input == 0u) {
        return Result<std::shared_ptr<ProgramState>>::fail(
            graph->status.reason());
      }
      inputs.push_back(input);
    }
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
  }
  const std::uint32_t output =
      graph_map(graph, inputs, map->expressions.front(), "service-free-map-dag",
                source->count);
  if (!graph->status || output == 0u) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        graph->status ? Reason::GraphBindingInvalid : graph->status.reason());
  }
  graph_output(graph, output);
  if (!graph->status) {
    return Result<std::shared_ptr<ProgramState>>::fail(graph->status.reason());
  }
  const std::array output_types{program->output_types.front()};
  return compile_graph(graph, program->input_types, output_types);
}

} // namespace rund::compute::detail::graph_compile
