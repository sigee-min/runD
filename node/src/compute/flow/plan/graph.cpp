#include "graph.hpp"

#include "../recipe.hpp"

#include "../../graph/build/indexed.hpp"
#include "../../graph/state.hpp"

#include <rund/compute/abi/graph.hpp>

#include <algorithm>
#include <limits>
#include <new>
#include <vector>

namespace rund::compute::detail {
namespace {

using GraphResult = Result<std::shared_ptr<GraphState>>;

[[nodiscard]] GraphResult fail(const Reason reason) {
  return GraphResult::fail(reason);
}

} // namespace

GraphResult materialize_graph(const std::shared_ptr<FlowState> &flow,
                              const std::shared_ptr<DeviceState> &device,
                              const std::span<const std::size_t> order,
                              const std::span<const MapRecipe> maps,
                              const std::span<const std::uint8_t> skipped) {
  if (flow == nullptr || maps.size() != flow->steps.size()) {
    return fail(Reason::GraphCapacity);
  }
  const auto graph = make_graph(device, "flow", flow_count(flow));
  if (graph == nullptr) {
    return fail(Reason::GraphCapacity);
  }
  std::vector<std::uint32_t> values(flow->values.size() + 1u, 0u);
  for (const std::uint32_t input : flow->inputs) {
    const FlowValue &value = flow->values[input - 1u];
    values[input] =
        graph_input_count(graph, value.type, value.count, value.fixed_format);
  }
  try {
    graph->bounded_inputs.reserve(flow->bounded_inputs.size());
    for (const BoundedInputSchema bounded : flow->bounded_inputs) {
      if (bounded.count == 0u || bounded.count >= values.size() ||
          values[bounded.count] == 0u) {
        return fail(Reason::BoundedCountInvalid);
      }
      graph->bounded_inputs.push_back(BoundedInputSchema{
          .count = values[bounded.count], .capacity = bounded.capacity});
    }
  } catch (const std::bad_alloc &) {
    return fail(Reason::GraphCapacity);
  }

  for (const std::size_t step_index : order) {
    if (!skipped.empty() && skipped[step_index] != 0u) {
      continue;
    }
    const FlowStep &step = flow->steps[step_index];
    if (std::holds_alternative<MapStep>(step)) {
      const MapRecipe &recipe = maps[step_index];
      if (recipe.fused) {
        continue;
      }
      std::vector<std::uint32_t> inputs;
      try {
        inputs.reserve(recipe.inputs.size());
        for (const std::uint32_t input : recipe.inputs) {
          if (input == 0u || input >= values.size() || values[input] == 0u) {
            return fail(Reason::GraphBindingInvalid);
          }
          inputs.push_back(values[input]);
        }
      } catch (const std::bad_alloc &) {
        return fail(Reason::GraphCapacity);
      }
      FlowControl control = recipe.control;
      control.count = control.count == 0u ? 0u : values[control.count];
      control.predicate =
          control.predicate == 0u ? 0u : values[control.predicate];
      ValueIds outputs;
      if (recipe.indices.empty()) {
        outputs = graph_map_multi_controlled(
            graph, inputs, recipe.expressions, control, recipe.name,
            flow->values[recipe.outputs.front() - 1u].count);
      } else {
        std::vector<std::uint32_t> indices;
        try {
          indices.reserve(recipe.indices.size());
          for (const std::uint32_t index : recipe.indices) {
            if (index == 0u) {
              indices.push_back(0u);
            } else if (index >= values.size() || values[index] == 0u) {
              return fail(Reason::GraphBindingInvalid);
            } else {
              indices.push_back(values[index]);
            }
          }
        } catch (const std::bad_alloc &) {
          return fail(Reason::GraphCapacity);
        }
        outputs = graph_map_indexed(
            graph, inputs, indices, recipe.expressions, control, recipe.name,
            flow->values[recipe.outputs.front() - 1u].count);
      }
      if (!graph->status) {
        return fail(graph->status.reason());
      }
      if (outputs.size() != recipe.outputs.size()) {
        return fail(Reason::GraphBindingInvalid);
      }
      for (std::size_t index = 0u; index < recipe.outputs.size(); ++index) {
        values[recipe.outputs[index]] = outputs[index];
      }
      continue;
    }

    if (const auto *const scan = std::get_if<ScanStep>(&step)) {
      FlowControl control = scan->control;
      control.count = control.count == 0u ? 0u : values[control.count];
      control.predicate =
          control.predicate == 0u ? 0u : values[control.predicate];
      values[scan->output] =
          graph_scan(graph, values[scan->input], scan->operation,
                     scan->count == 0u ? 0u : values[scan->count], control);
      if (!graph->status) {
        return fail(graph->status.reason());
      }
      continue;
    }

    const auto &primitive = std::get<FlowPrimitive>(step);
    const std::span<const std::uint32_t> inputs =
        flow->value_ids.view(primitive.inputs);
    const std::span<const std::uint32_t> outputs =
        flow->value_ids.view(primitive.outputs);
    std::vector<GraphArg> arguments;
    std::vector<GraphArg> recipes;
    try {
      arguments.reserve(inputs.size());
      for (const std::uint32_t input : inputs) {
        const FlowValue &value = flow->values[input - 1u];
        arguments.push_back(GraphArg{values[input], value.type, value.count,
                                     value.fixed_format});
      }
      if (flow_count(flow) == 0u) {
        recipes.reserve(outputs.size() +
                        (primitive.operation == Primitive::Sort ||
                                 primitive.operation == Primitive::Argsort
                             ? 1u
                             : 0u));
        if (primitive.operation == Primitive::Sort ||
            primitive.operation == Primitive::Argsort) {
          const FlowValue &keys = flow->values[inputs.front() - 1u];
          recipes.push_back(
              GraphArg{0u, keys.type, keys.count, keys.fixed_format});
          recipes.push_back(GraphArg{0u, Type::U32, keys.count, FixedFormat{}});
        } else {
          for (const std::uint32_t output : outputs) {
            const FlowValue &value = flow->values[output - 1u];
            recipes.push_back(
                GraphArg{0u, value.type, value.count, value.fixed_format});
          }
        }
      }
    } catch (const std::bad_alloc &) {
      return fail(Reason::GraphCapacity);
    }
    FlowControl control = primitive.control;
    control.count = control.count == 0u ? 0u : values[control.count];
    control.predicate =
        control.predicate == 0u ? 0u : values[control.predicate];
    GraphOut output = graph_primitive(graph, primitive.operation, arguments,
                                      primitive.options, recipes, control);
    if (!graph->status) {
      return fail(graph->status.reason());
    }
    if (outputs.size() == 1u) {
      const std::uint32_t primary = outputs.front();
      values[primary] = output.value;
      const FlowValue recipe = flow->values[primary - 1u];
      flow->values[primary - 1u] =
          FlowValue{.type = output.type,
                    .fixed_format = output.fixed_format,
                    .count = output.count,
                    .guard = recipe.guard,
                    .active = recipe.active,
                    .parent = recipe.parent};
      continue;
    }
    if (outputs.size() != output.outputs.size()) {
      return fail(Reason::GraphBindingInvalid);
    }
    for (std::size_t index = 0u; index < outputs.size(); ++index) {
      const std::uint32_t recipe_value = outputs[index];
      const GraphArg &value = output.outputs[index];
      values[recipe_value] = value.value;
      const FlowValue recipe = flow->values[recipe_value - 1u];
      flow->values[recipe_value - 1u] =
          FlowValue{.type = value.type,
                    .fixed_format = value.fixed_format,
                    .count = value.count,
                    .guard = recipe.guard,
                    .active = recipe.active,
                    .parent = recipe.parent};
    }
  }

  for (std::size_t index = 0u; index < flow->values.size(); ++index) {
    const FlowValue &recipe = flow->values[index];
    const std::uint32_t compiled = values[index + 1u];
    if (compiled == 0u) {
      continue;
    }
    if (compiled > graph->values.size()) {
      return fail(Reason::GraphBindingInvalid);
    }
    GraphValue &value = graph->values[compiled - 1u];
    if (recipe.active != 0u) {
      if (recipe.active >= values.size() || values[recipe.active] == 0u) {
        return fail(Reason::BoundedCountInvalid);
      }
      value.active = values[recipe.active];
    }
    if (recipe.parent != 0u) {
      if (recipe.parent >= values.size() || values[recipe.parent] == 0u) {
        return fail(Reason::BoundedCountInvalid);
      }
      value.parent = values[recipe.parent];
    }
    if (value.active == compiled || value.parent == compiled) {
      return fail(Reason::BoundedCountInvalid);
    }
  }

  std::vector<std::uint32_t> selected;
  try {
    if (flow->outputs.empty()) {
      selected.push_back(values[flow->output]);
    } else {
      selected.reserve(flow->outputs.size());
      for (const std::uint32_t output : flow->outputs) {
        selected.push_back(values[output]);
      }
    }
  } catch (const std::bad_alloc &) {
    return fail(Reason::GraphCapacity);
  }
  graph_outputs(graph, selected);
  if (!graph->status) {
    return fail(graph->status.reason());
  }

  if (!flow->logical_outputs.empty()) {
    const auto &projection = graph->identity_outputs.empty()
                                 ? graph->outputs
                                 : graph->identity_outputs;
    if (projection.size() != selected.size()) {
      return fail(Reason::GraphBindingInvalid);
    }
    std::vector<std::uint32_t> identities;
    try {
      identities.reserve(flow->logical_outputs.size());
      for (const std::uint32_t output : flow->logical_outputs) {
        const auto found =
            std::find(flow->outputs.begin(), flow->outputs.end(), output);
        if (found == flow->outputs.end()) {
          return fail(Reason::GraphBindingInvalid);
        }
        const std::size_t index =
            static_cast<std::size_t>(found - flow->outputs.begin());
        identities.push_back(projection[index]);
      }
    } catch (const std::bad_alloc &) {
      return fail(Reason::GraphCapacity);
    }
    graph_identity_outputs(graph, identities);
    if (!graph->status) {
      return fail(graph->status.reason());
    }
  }
  return GraphResult::success(graph);
}

} // namespace rund::compute::detail
