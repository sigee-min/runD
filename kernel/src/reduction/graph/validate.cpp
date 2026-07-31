#include "validate/local.hpp"

namespace rund::kernel::reduction::graph {

FoldGraphValidationResult ValidateBuiltFixedBinaryTreeGraph(
    const FoldGraph& graph,
    const FoldOperation operation,
    const FoldPrimitiveSpec& primitive,
    const u32 partition_count,
    const u32 required_nodes,
    const u32 required_edges,
    const u32 required_scratch_slots) {
  const FoldGraphValidationResult shape =
      validate_detail::ValidateFixedGraphShape(graph,
                                               primitive,
                                               partition_count,
                                               required_nodes,
                                               required_edges,
                                               required_scratch_slots);
  if (!shape.ok) {
    return shape;
  }
  const FoldGraphValidationResult nodes =
      validate_detail::ValidateFixedGraphPartitionNodes(graph,
                                                       operation,
                                                       primitive,
                                                       partition_count,
                                                       required_scratch_slots);
  if (!nodes.ok) {
    return nodes;
  }
  return validate_detail::ValidateFixedGraphReductionEdges(graph,
                                                          operation,
                                                          primitive,
                                                          partition_count,
                                                          required_edges,
                                                          required_scratch_slots);
}

} // namespace rund::kernel::reduction::graph
