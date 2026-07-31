#include "model.hpp"

#include "../local.hpp"

#include <rund/compute/abi/graph.hpp>

#include <memory>
#include <string>

namespace rund::compute::detail {

std::shared_ptr<GraphState>
make_graph(const std::shared_ptr<DeviceState> &device,
           const std::string_view name, const std::size_t count) {
  try {
    auto graph = std::make_shared<GraphState>();
    graph->device = device;
    graph->name = std::string{name};
    graph->count = count;
    if (device == nullptr) {
      graph_detail::reject(*graph, Reason::DeviceInvalid);
    } else if (name.empty()) {
      graph_detail::reject(*graph, Reason::NameEmpty);
    }
    return graph;
  } catch (const std::bad_alloc &) {
    return {};
  }
}

std::uint32_t graph_input(const std::shared_ptr<GraphState> &graph,
                          const Type type) {
  return graph_input_count(graph, type, graph == nullptr ? 0u : graph->count);
}

std::uint32_t graph_input_count(const std::shared_ptr<GraphState> &graph,
                                const Type type, const std::size_t count,
                                const FixedFormat fixed_format) {
  if (graph == nullptr) {
    return 0u;
  }
  const std::uint32_t value =
      graph_detail::append(*graph, type, count, fixed_format);
  if (value != 0u) {
    try {
      graph->inputs.push_back(value);
    } catch (const std::bad_alloc &) {
      graph_detail::reject(*graph, Reason::GraphCapacity);
      return 0u;
    }
  }
  return value;
}

void reject_graph(const std::shared_ptr<GraphState> &graph,
                  const Reason reason) {
  if (graph != nullptr) {
    graph_detail::reject(*graph, reason);
  }
}

} // namespace rund::compute::detail
