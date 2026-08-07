#include "indexed.hpp"
#include "model.hpp"

#include "../../fixed/format.hpp"
#include "../../type.hpp"
#include "../local.hpp"
#include "index.hpp"

#include <rund/compute/abi/graph.hpp>

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace graph_build_detail {

bool append_map(GraphState &graph, const std::string_view name,
                const std::span<const std::uint32_t> inputs,
                const std::span<const std::uint32_t> outputs,
                const std::span<const ExprRef> expressions,
                const FlowControl control,
                const std::span<const MapRead> reads) {
  try {
    const bool published =
        graph.value_ids.publish(inputs, outputs, [&](const ValueRoutes routes) {
          graph.steps.emplace_back(MapStep{
              .name = std::string{name},
              .inputs = routes.inputs,
              .outputs = routes.outputs,
              .expressions =
                  std::vector<ExprRef>{expressions.begin(), expressions.end()},
              .reads = std::vector<MapRead>{reads.begin(), reads.end()},
              .control = control,
          });
        });
    if (!published) {
      graph_detail::reject(graph, Reason::GraphCapacity);
      return false;
    }
    return true;
  } catch (const std::bad_alloc &) {
    graph_detail::reject(graph, Reason::GraphCapacity);
    return false;
  }
}

namespace {

[[nodiscard]] std::optional<std::uint32_t>
direct_input(const GraphState &graph,
             const std::span<const std::uint32_t> inputs,
             const std::span<const ExprRef> expressions) noexcept {
  if (expressions.size() != 1u) {
    return std::nullopt;
  }
  const ExprRef &expression = expressions.front();
  if (expression.state == nullptr || expression.node == 0u ||
      expression.node > expression.state->nodes.size()) {
    return std::nullopt;
  }
  const ExprNode &root = expression.state->nodes[expression.node - 1u];
  if (root.operation != ExprOp::Input || root.left >= inputs.size() ||
      root.type != expression.type ||
      root.fixed_format != expression.fixed_format) {
    return std::nullopt;
  }
  const std::uint32_t value = inputs[root.left];
  if (!graph_detail::valid(graph, value)) {
    return std::nullopt;
  }
  const GraphValue &shape = graph.values[value - 1u];
  if (shape.type != expression.type ||
      shape.fixed_format != expression.fixed_format) {
    return std::nullopt;
  }
  return value;
}

} // namespace

ValueIds build_map(const std::shared_ptr<GraphState> &graph,
                   const std::span<const std::uint32_t> sources,
                   const std::span<const std::uint32_t> indices,
                   const std::span<const ExprRef> expressions,
                   const FlowControl control, const std::string_view name,
                   const std::size_t output_count) {
  if (graph == nullptr) {
    return {};
  }
  const bool indexed = !indices.empty();
  if (sources.size() > MaxMapInputs ||
      (indexed && indices.size() != sources.size()) || expressions.empty() ||
      expressions.size() > MaxOutputs) {
    graph_detail::reject(*graph, Reason::GraphValueInvalid);
    return {};
  }
  GraphValue shape{.type = expressions.front().type,
                   .fixed_format = expressions.front().fixed_format,
                   .count = output_count};
  if (!sources.empty()) {
    const std::uint32_t first = sources.front();
    if (!graph_detail::valid(*graph, first)) {
      graph_detail::reject(*graph, Reason::GraphValueInvalid);
      return {};
    }
    if (!indexed && shape.count == 0u) {
      shape = graph->values[first - 1u];
    } else if (shape.count == 0u) {
      graph_detail::reject(*graph, Reason::GraphShapeMismatch);
      return {};
    }
  } else if (output_count == 0u && graph->count != 0u) {
    graph_detail::reject(*graph, Reason::GraphShapeMismatch);
    return {};
  }

  std::vector<std::uint32_t> inputs{sources.begin(), sources.end()};
  std::vector<MapRead> reads;
  if (indexed) {
    reads.resize(sources.size());
  }
  for (std::size_t source = 0u; source < sources.size(); ++source) {
    const std::uint32_t value = sources[source];
    if (!graph_detail::valid(*graph, value)) {
      graph_detail::reject(*graph, Reason::GraphValueInvalid);
      return {};
    }
    const GraphValue source_shape = graph->values[value - 1u];
    const std::uint32_t index = indexed ? indices[source] : 0u;
    if (index == 0u) {
      if (source_shape.count == shape.count) {
        continue;
      }
      if (source_shape.count == 1u && shape.count != 1u) {
        if (reads.empty()) {
          reads.resize(sources.size());
        }
        reads[source] = MapRead{.mode = MapReadMode::Uniform};
      } else {
        graph_detail::reject(*graph, Reason::GraphShapeMismatch);
        return {};
      }
      continue;
    }
    if (!graph_detail::valid(*graph, index) ||
        graph->values[index - 1u].type != Type::U32 ||
        graph->values[index - 1u].count != shape.count ||
        source_shape.count == 0u ||
        source_shape.count > std::numeric_limits<std::uint32_t>::max()) {
      graph_detail::reject(*graph, Reason::GraphShapeMismatch);
      return {};
    }
    const auto found =
        std::find(inputs.begin() + static_cast<std::ptrdiff_t>(sources.size()),
                  inputs.end(), index);
    std::size_t binding = 0u;
    if (found == inputs.end()) {
      binding = inputs.size();
      inputs.push_back(index);
    } else {
      binding = static_cast<std::size_t>(found - inputs.begin());
    }
    reads[source] =
        MapRead{.mode = MapReadMode::Indexed,
                .index = static_cast<std::uint32_t>(binding),
                .count = static_cast<std::uint32_t>(source_shape.count)};
  }

  const auto valid_control_value = [&](const std::uint32_t value) {
    return graph_detail::valid(*graph, value) &&
           graph->values[value - 1u].count == 1u &&
           (graph->values[value - 1u].type == Type::U32 ||
            graph->values[value - 1u].type == Type::U64);
  };
  if ((control.count != 0u &&
       (!valid_control_value(control.count) || control.capacity == 0u ||
        control.capacity != shape.count)) ||
      (control.count == 0u && control.capacity != 0u) ||
      (control.predicate != 0u &&
       (!valid_control_value(control.predicate) ||
        (graph->values[control.predicate - 1u].type == Type::U32 &&
         control.predicate_expected >
             std::numeric_limits<std::uint32_t>::max()))) ||
      (control.predicate == 0u &&
       (control.predicate_expected != 0u ||
        (control.iteration != 0u && control.count == 0u)))) {
    graph_detail::reject(*graph, Reason::BoundedCountInvalid);
    return {};
  }

  graph_build_detail::Index<const ExprState *, MaxOutputs> states;
  for (const ExprRef &expression : expressions) {
    if (expression.state == nullptr || !expression.state->status ||
        expression.state->nodes.empty() ||
        (!sources.empty() &&
         type_bytes(expression.type) !=
             type_bytes(graph->values[sources.front() - 1u].type) &&
         !(expressions.size() == 1u &&
           is_width_mask(expression,
                         graph->values[sources.front() - 1u].type)))) {
      graph_detail::reject(*graph, Reason::GraphTypeMismatch);
      return {};
    }
    if (!states.add(expression.state.get())) {
      continue;
    }
    for (const ExprNode &node : expression.state->nodes) {
      if (node.operation != ExprOp::Input) {
        continue;
      }
      if (sources.empty() || node.left >= sources.size() ||
          node.type != graph->values[sources[node.left] - 1u].type) {
        graph_detail::reject(*graph, Reason::GraphTypeMismatch);
        return {};
      }
    }
  }
  if (!indexed && control.empty()) {
    if (const auto value =
            graph_build_detail::direct_input(*graph, sources, expressions);
        value && graph->values[*value - 1u].count == shape.count) {
      ValueIds output;
      output.push_back(*value);
      return output;
    }
  }
  if (graph->steps.size() >= graph_detail::MaxSteps) {
    graph_detail::reject(*graph, Reason::GraphCapacity);
    return {};
  }

  ValueIds outputs;
  for (const ExprRef &expression : expressions) {
    const std::uint32_t output = graph_detail::append(
        *graph, expression.type, shape.count, expression.fixed_format);
    if (output == 0u) {
      return {};
    }
    outputs.push_back(output);
  }
  if (!graph_build_detail::append_map(*graph, name, inputs, outputs,
                                      expressions, control, reads)) {
    return {};
  }
  return outputs;
}

} // namespace graph_build_detail

std::uint32_t graph_map(const std::shared_ptr<GraphState> &graph,
                        const std::span<const std::uint32_t> inputs,
                        const ExprRef expression, const std::string_view name,
                        const std::size_t output_count) {
  const std::array expressions{expression};
  const ValueIds outputs =
      graph_map_multi(graph, inputs, expressions, name, output_count);
  return outputs.size() == 1u ? outputs.front() : 0u;
}

ValueIds graph_map_multi(const std::shared_ptr<GraphState> &graph,
                         const std::span<const std::uint32_t> inputs,
                         const std::span<const ExprRef> expressions,
                         const std::string_view name,
                         const std::size_t output_count) {
  return graph_map_multi_controlled(graph, inputs, expressions, {}, name,
                                    output_count);
}

ValueIds graph_map_multi_controlled(const std::shared_ptr<GraphState> &graph,
                                    const std::span<const std::uint32_t> inputs,
                                    const std::span<const ExprRef> expressions,
                                    const FlowControl control,
                                    const std::string_view name,
                                    const std::size_t output_count) {
  return graph_build_detail::build_map(graph, inputs, {}, expressions, control,
                                       name, output_count);
}

ValueIds graph_map_indexed(const std::shared_ptr<GraphState> &graph,
                           const std::span<const std::uint32_t> sources,
                           const std::span<const std::uint32_t> indices,
                           const std::span<const ExprRef> expressions,
                           const FlowControl control,
                           const std::string_view name,
                           const std::size_t output_count) {
  return graph_build_detail::build_map(graph, sources, indices, expressions,
                                       control, name, output_count);
}

} // namespace rund::compute::detail
