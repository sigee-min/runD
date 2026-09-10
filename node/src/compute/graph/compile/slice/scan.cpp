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

[[nodiscard]] bool plain_u64(const GraphState &graph, const std::uint32_t value,
                             const std::size_t count) noexcept {
  return value != 0u && value <= graph.values.size() &&
         graph.values[value - 1u].type == Type::U64 &&
         graph.values[value - 1u].fixed_format == FixedFormat{} &&
         graph.values[value - 1u].count == count;
}

[[nodiscard]] const ScanStep *
exact_terminal_scan(const ProgramState &program) noexcept {
  const GraphState *const graph = program.canonical_graph.get();
  if (graph == nullptr || !graph->status || graph->device != program.device ||
      graph->steps.size() < 2u || graph->inputs.empty() ||
      graph->inputs.size() != program.input_types.size() ||
      graph->inputs.size() > ServiceFreeMapScanInputCapacity ||
      graph->outputs.size() != 1u || program.input_types.empty() ||
      program.output_types.size() != 1u ||
      program.input_sizes.size() != program.input_types.size() ||
      program.output_sizes.size() != 1u ||
      program.input_formats.size() != program.input_types.size() ||
      program.output_formats.size() != 1u ||
      !std::all_of(program.input_types.begin(), program.input_types.end(),
                   [](const Type type) { return type == Type::U64; }) ||
      program.output_types.front() != Type::U64 ||
      !std::all_of(
          program.input_formats.begin(), program.input_formats.end(),
          [](const FixedFormat format) { return format == FixedFormat{}; }) ||
      program.output_formats.front() != FixedFormat{} ||
      program.input_sizes.front() == 0u ||
      !std::all_of(program.input_sizes.begin(), program.input_sizes.end(),
                   [&program](const std::size_t count) {
                     return count == program.input_sizes.front();
                   }) ||
      program.input_sizes.front() != program.output_sizes.front() ||
      graph->count != program.input_sizes.front() ||
      !std::all_of(graph->inputs.begin(), graph->inputs.end(),
                   [graph](const std::uint32_t input) {
                     return plain_u64(*graph, input, graph->count);
                   }) ||
      !std::all_of(graph->steps.begin(), graph->steps.end() - 1u,
                   [](const GraphStep &step) {
                     return std::holds_alternative<MapStep>(step);
                   })) {
    return nullptr;
  }
  const auto *const scan = std::get_if<ScanStep>(&graph->steps.back());
  return scan != nullptr && scan->count == 0u && scan->control.empty() &&
                 scan->control.iteration == 0u &&
                 plain_u64(*graph, scan->input, graph->count) &&
                 plain_u64(*graph, scan->output, graph->count) &&
                 scan->output == graph->outputs.front() &&
                 (scan->operation == Scan::InclusiveSum ||
                  scan->operation == Scan::ExclusiveSum)
             ? scan
             : nullptr;
}

} // namespace

Result<std::shared_ptr<ProgramState>>
compile_service_free_map_scan(const std::shared_ptr<ProgramState> &program) {
  const ScanStep *const scan =
      program == nullptr ? nullptr : exact_terminal_scan(*program);
  if (scan == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        Reason::PrimitiveUnsupported);
  }
  const GraphState &source = *program->canonical_graph;
  auto semantic = compile_service_free_map_semantic_inputs(
      source, Type::U64, source.count, source.inputs, scan->input);
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
      composed.value_ids.view(map->inputs).size() != source.inputs.size()) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::ProgramInvalid);
  }

  const auto graph =
      make_graph(program->device, "service-free-map-scan", source.count);
  if (graph == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
  }
  std::vector<std::uint32_t> inputs;
  try {
    inputs.reserve(source.inputs.size());
    for (std::size_t index = 0u; index < source.inputs.size(); ++index) {
      const std::uint32_t input =
          graph_input_count(graph, program->input_types[index], source.count);
      if (input == 0u) {
        return Result<std::shared_ptr<ProgramState>>::fail(
            graph->status.reason());
      }
      inputs.push_back(input);
    }
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
  }
  const std::uint32_t mapped =
      graph_map(graph, inputs, map->expressions.front(), "service-free-map-dag",
                source.count);
  const std::uint32_t output =
      mapped == 0u ? 0u : graph_scan(graph, mapped, scan->operation);
  if (!graph->status || output == 0u) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        graph->status ? Reason::GraphBindingInvalid : graph->status.reason());
  }
  graph_output(graph, output);
  if (!graph->status) {
    return Result<std::shared_ptr<ProgramState>>::fail(graph->status.reason());
  }
  const std::array output_types{Type::U64};
  return compile_graph(graph, program->input_types, output_types);
}

} // namespace rund::compute::detail::graph_compile
