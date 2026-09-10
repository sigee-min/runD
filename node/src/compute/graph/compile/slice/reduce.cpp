#include "internal.hpp"

#include <rund/compute/abi/graph.hpp>

#include <array>

namespace rund::compute::detail::graph_compile::slice_detail {

Result<std::shared_ptr<ProgramState>> compile_reduce(const SliceSource source) {
  const auto graph =
      make_graph(source.graph->device, "tiled-graph-reduce", source.capacity);
  if (graph == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
  }
  const std::uint32_t value =
      graph_input_count(graph, source.type, source.capacity);
  const std::uint32_t count = graph_input_count(graph, Type::U64, 1u);
  if (value == 0u || count == 0u ||
      !bind_count(*graph, count, source.capacity)) {
    return Result<std::shared_ptr<ProgramState>>::fail(graph->status.reason());
  }
  const std::array inputs{
      GraphArg{value, source.type, source.capacity, FixedFormat{}},
      GraphArg{count, Type::U64, 1u, FixedFormat{}},
  };
  const GraphOut output =
      graph_primitive(graph, Primitive::Reduce, inputs, source.reduce->options);
  if (!graph->status || output.value == 0u || output.type != source.type ||
      output.count != 1u || output.fixed_format != FixedFormat{}) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        graph->status ? Reason::GraphBindingInvalid : graph->status.reason());
  }
  graph_output(graph, output.value);
  const std::array input_types{source.type, Type::U64};
  const std::array output_types{source.type};
  return compile_graph(graph, input_types, output_types);
}

} // namespace rund::compute::detail::graph_compile::slice_detail
