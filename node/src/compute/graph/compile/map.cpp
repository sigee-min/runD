#include "local.hpp"

#include "../../cpu/graph.hpp"
#include "../../map/build.hpp"
#include "../../map/name.hpp"
#include "../local.hpp"

#include <accel/graph/factory.hpp>

#include <array>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rund::compute::detail::graph_compile {
namespace {

[[nodiscard]] kernel::GraphControlSource
control_source(const GraphState &graph, const std::uint32_t value) noexcept {
  return graph.values[value - 1u].type == Type::U64
             ? kernel::GraphControlSource::U64
             : kernel::GraphControlSource::U32;
}

[[nodiscard]] kernel::GraphControl control(const GraphState &graph,
                                           const MapStep &map,
                                           const std::size_t input_count,
                                           const std::size_t output_count) {
  kernel::GraphControl lowered{
      .capacity = map.control.capacity,
      .predicate_expected = map.control.predicate_expected,
      .iteration = map.control.iteration,
  };
  std::size_t binding = input_count + output_count;
  if (map.control.count != 0u) {
    lowered.count_source = control_source(graph, map.control.count);
    lowered.count_binding = static_cast<std::uint32_t>(binding++);
  }
  if (map.control.predicate != 0u) {
    lowered.predicate_source = control_source(graph, map.control.predicate);
    lowered.predicate_binding = static_cast<std::uint32_t>(binding++);
  }
  return lowered;
}

} // namespace

Status map(Lowering &lowering, const std::size_t index, const MapStep &step) {
  if (!lowering.graph->value_ids.valid(step.inputs) ||
      !lowering.graph->value_ids.valid(step.outputs)) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  const std::span<const std::uint32_t> inputs =
      lowering.graph->value_ids.view(step.inputs);
  const std::span<const std::uint32_t> outputs =
      lowering.graph->value_ids.view(step.outputs);
  if (outputs.empty() || lowering.operation >= lowering.operations.size()) {
    return Status::fail(outputs.empty() ? Reason::GraphBindingInvalid
                                        : Reason::GraphOperationInvalid);
  }
  std::vector<Type> input_types;
  std::vector<Type> output_types;
  try {
    input_types.reserve(inputs.size());
    for (const std::uint32_t input : inputs) {
      input_types.push_back(lowering.graph->values[input - 1u].type);
    }
    output_types.reserve(outputs.size());
    for (const std::uint32_t output : outputs) {
      output_types.push_back(lowering.graph->values[output - 1u].type);
    }
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }

  compute_dsl::ComputeOp &operation = lowering.operations[lowering.operation++];
  const kernel::GraphControl execution =
      control(*lowering.graph, step, inputs.size(), outputs.size());
  const std::size_t count = lowering.graph->values[outputs.front() - 1u].count;
  if (lowering.cpu()) {
    if (!execution.valid(lowering.cpu_refs.back().size())) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    auto prepared = prepare_cpu_map(lowering.graph->device, count, output_types,
                                    input_types, std::move(operation));
    if (!prepared) {
      return Status::fail(prepared.reason());
    }
    lowering.program->cpu_graph->maps[index] = std::move(prepared).value();
    const kernel::ComputeMap &stored =
        lowering.program->cpu_graph->maps[index]->map;
    const auto &refs = lowering.cpu_refs.back();
    lowering.cpu_nodes.push_back(kernel::GraphNode{
        .op_hash_hi = stored.op_hash_hi,
        .op_hash_lo = stored.op_hash_lo,
        .buffers = refs.data(),
        .buffer_count = refs.size(),
        .kind = kernel::NodeKind::Map,
        .element_count = count,
        .control = execution,
    });
    lowering.program->cpu_graph->runtime->steps.emplace_back(CpuRuntimeMap{
        .inputs = std::vector<std::uint32_t>{inputs.begin(), inputs.end()},
        .outputs = std::vector<std::uint32_t>{outputs.begin(), outputs.end()},
        .control = step.control,
    });
    return Status::success();
  }

  try {
    lowering.accel_refs.emplace_back();
    auto &refs = lowering.accel_refs.back();
    refs.reserve(inputs.size() + outputs.size() + 2u);
    for (std::size_t port = 0u; port < inputs.size(); ++port) {
      const auto input =
          resident(lowering, index, inputs[port], kernel::BufferRole::Read,
                   map_input_name(inputs.size(), port));
      if (!input) {
        return Status::fail(Reason::AccelProgramInvalid);
      }
      refs.push_back(*input);
    }
    for (std::size_t port = 0u; port < outputs.size(); ++port) {
      const auto output =
          resident(lowering, index, outputs[port], kernel::BufferRole::Write,
                   map_output_name(outputs.size(), port));
      if (!output) {
        return Status::fail(Reason::AccelProgramInvalid);
      }
      refs.push_back(*output);
    }
    if (step.control.count != 0u) {
      const auto count_ref =
          resident(lowering, index, step.control.count,
                   kernel::BufferRole::Read, "logical_count");
      if (!count_ref) {
        return Status::fail(Reason::AccelProgramInvalid);
      }
      refs.push_back(*count_ref);
    }
    if (step.control.predicate != 0u) {
      const auto predicate = resident(lowering, index, step.control.predicate,
                                      kernel::BufferRole::Read, "predicate");
      if (!predicate) {
        return Status::fail(Reason::AccelProgramInvalid);
      }
      refs.push_back(*predicate);
    }
    if (!execution.valid(refs.size())) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    auto node = rund::AccelMap(operation.ir(), refs.data(), refs.size(), count);
    node.control = execution;
    node.barrier_before = lowering.barriers[index] != 0u;
    lowering.accel_nodes.push_back(std::move(node));
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_compile
