#include "model.hpp"

#include "../local.hpp"
#include "index.hpp"

#include <rund/compute/abi/graph.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <span>

namespace rund::compute::detail {
namespace graph_build_detail {

std::uint32_t materialize(GraphState &graph, const std::uint32_t source,
                          const std::string_view name) {
  if (!graph_detail::valid(graph, source)) {
    graph_detail::reject(graph, Reason::GraphValueInvalid);
    return 0u;
  }
  if (graph.steps.size() >= graph_detail::MaxSteps) {
    graph_detail::reject(graph, Reason::GraphCapacity);
    return 0u;
  }
  const GraphValue &value = graph.values[source - 1u];
  const auto expression = make_expr();
  const ExprRef copy = input(expression, value.type, 0u, value.fixed_format);
  if (copy.state == nullptr || !copy.state->status) {
    graph_detail::reject(graph, Reason::ExpressionCapacity);
    return 0u;
  }
  const std::uint32_t output =
      graph_detail::append(graph, value.type, value.count, value.fixed_format);
  if (output == 0u) {
    return 0u;
  }
  const std::array inputs{source};
  const std::array outputs{output};
  const std::array expressions{copy};
  return append_map(graph, name, inputs, outputs, expressions) ? output : 0u;
}

} // namespace graph_build_detail

void graph_output(const std::shared_ptr<GraphState> &graph,
                  const std::uint32_t output) {
  const std::array outputs{output};
  graph_outputs(graph, outputs);
}

void graph_outputs(const std::shared_ptr<GraphState> &graph,
                   const std::span<const std::uint32_t> outputs) {
  if (graph == nullptr) {
    return;
  }
  if (outputs.empty() || outputs.size() > MaxOutputs) {
    graph_detail::reject(*graph, Reason::GraphOutputCapacity);
    return;
  }
  if (!graph->outputs.empty()) {
    graph_detail::reject(*graph, Reason::GraphOutputDuplicate);
    return;
  }
  for (const std::uint32_t output : outputs) {
    if (!graph_detail::valid(*graph, output)) {
      graph_detail::reject(*graph, Reason::GraphValueInvalid);
      return;
    }
  }

  try {
    graph->outputs.reserve(outputs.size());
  } catch (const std::bad_alloc &) {
    graph_detail::reject(*graph, Reason::GraphCapacity);
    return;
  }
  graph_build_detail::Index<std::uint32_t, MaxOutputs> sources;
  std::array<std::uint32_t, MaxOutputs> projection{};
  std::size_t projection_count = 0u;
  for (const std::uint32_t output : outputs) {
    const auto [ordinal, inserted] =
        sources.admit(output, graph->outputs.size());
    if (!inserted) {
      projection[projection_count++] = graph->outputs[ordinal];
      continue;
    }

    std::uint32_t selected = output;
    // graph_input_count appends monotonically increasing value IDs, so ordered
    // interface identity is also an exact allocation-free membership index.
    if (std::binary_search(graph->inputs.begin(), graph->inputs.end(),
                           output)) {
      selected =
          graph_build_detail::materialize(*graph, output, "graph-output");
      if (selected == 0u) {
        return;
      }
    }
    try {
      graph->outputs.push_back(selected);
    } catch (const std::bad_alloc &) {
      graph_detail::reject(*graph, Reason::GraphCapacity);
      return;
    }
    projection[projection_count++] = selected;
  }

  const bool direct =
      projection_count == graph->outputs.size() &&
      std::equal(projection.begin(), projection.begin() + projection_count,
                 graph->outputs.begin());
  if (!direct) {
    try {
      graph->identity_outputs.assign(projection.begin(),
                                     projection.begin() + projection_count);
    } catch (const std::bad_alloc &) {
      graph_detail::reject(*graph, Reason::GraphCapacity);
    }
  }
}

void graph_identity_outputs(const std::shared_ptr<GraphState> &graph,
                            const std::span<const std::uint32_t> outputs) {
  if (graph == nullptr || !graph->status) {
    return;
  }
  if (outputs.empty() || outputs.size() > MaxOutputs) {
    graph_detail::reject(*graph, Reason::GraphOutputCapacity);
    return;
  }
  graph_build_detail::Index<std::uint32_t, MaxOutputs> admitted;
  for (const std::uint32_t output : graph->outputs) {
    static_cast<void>(admitted.add(output));
  }
  for (const std::uint32_t output : outputs) {
    if (!admitted.find(output)) {
      graph_detail::reject(*graph, Reason::GraphValueInvalid);
      return;
    }
  }
  if (outputs.size() == graph->outputs.size() &&
      std::equal(outputs.begin(), outputs.end(), graph->outputs.begin())) {
    graph->identity_outputs.clear();
    return;
  }
  try {
    graph->identity_outputs.assign(outputs.begin(), outputs.end());
  } catch (const std::bad_alloc &) {
    graph_detail::reject(*graph, Reason::GraphCapacity);
  }
}

} // namespace rund::compute::detail
