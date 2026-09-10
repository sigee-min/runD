#include "internal.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rund::compute::detail::graph_compile::slice_detail {

void append_referenced_resources(TiledGraphSlices &result,
                                 const GraphState &graph) {
  std::vector<bool> referenced(graph.values.size() + 1u, false);
  for (const TiledGraphStageSlice &stage : result.stages) {
    for (const std::uint32_t value : stage.inputs) {
      referenced[value] = true;
    }
    for (const std::uint32_t value : stage.outputs) {
      referenced[value] = true;
    }
  }
  for (std::uint32_t value = 1u; value <= graph.values.size(); ++value) {
    if (!referenced[value]) {
      continue;
    }
    const GraphValue &shape = graph.values[value - 1u];
    const bool external_input =
        std::find(result.input_resources.begin(), result.input_resources.end(),
                  value) != result.input_resources.end();
    result.resources.push_back(TiledGraphSliceResource{
        .resource = value,
        .type = shape.type,
        .format = shape.fixed_format,
        .count = shape.count,
        .kind = external_input ? SliceResourceKind::ExternalInput
                : value == result.output_resource
                    ? SliceResourceKind::ExternalOutput
                    : SliceResourceKind::Internal,
    });
  }
}

} // namespace rund::compute::detail::graph_compile::slice_detail
