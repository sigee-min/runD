#include "model.hpp"

#include "../local.hpp"

#include <memory>

namespace rund::compute::detail {

std::uint32_t graph_scan(const std::shared_ptr<GraphState> &graph,
                         const std::uint32_t input, const Scan operation,
                         const std::uint32_t count, const FlowControl control) {
  if (graph == nullptr) {
    return 0u;
  }
  if (!graph_detail::scan_operation(operation)) {
    graph_detail::reject(*graph, Reason::ScanOpUnsupported);
    return 0u;
  }
  if (!graph_detail::valid(*graph, input)) {
    graph_detail::reject(*graph, Reason::GraphValueInvalid);
    return 0u;
  }
  if (count != 0u) {
    if (!graph_detail::valid(*graph, count)) {
      graph_detail::reject(*graph, Reason::GraphValueInvalid);
      return 0u;
    }
    const GraphValue &logical = graph->values[count - 1u];
    if (logical.count != 1u ||
        (logical.type != Type::U32 && logical.type != Type::U64)) {
      graph_detail::reject(*graph, Reason::BoundedCountInvalid);
      return 0u;
    }
  }
  if ((!control.empty() || control.iteration != 0u) &&
      (count == 0u || control.count != count || control.predicate != 0u ||
       control.capacity != graph->values[input - 1u].count ||
       control.iteration == 0u)) {
    graph_detail::reject(*graph, Reason::BoundedCountInvalid);
    return 0u;
  }
  if (graph->steps.size() >= graph_detail::MaxSteps) {
    graph_detail::reject(*graph, Reason::GraphCapacity);
    return 0u;
  }
  const GraphValue &source = graph->values[input - 1u];
  const std::uint32_t output = graph_detail::append(
      *graph, source.type, source.count, source.fixed_format);
  if (output == 0u) {
    return 0u;
  }
  try {
    graph->steps.push_back(ScanStep{input, output, count, operation, control});
  } catch (const std::bad_alloc &) {
    graph_detail::reject(*graph, Reason::GraphCapacity);
    return 0u;
  }
  return output;
}

} // namespace rund::compute::detail
