#include "local.hpp"

#include <cstddef>

namespace rund::kernel::reduction::graph::validate_detail {

FoldGraphValidationResult ValidateFixedGraphShape(const FoldGraph& graph,
                                                  const FoldPrimitiveSpec& primitive,
                                                  const u32 partition_count,
                                                  const u32 required_nodes,
                                                  const u32 required_edges,
                                                  const u32 required_scratch_slots) {
  if (!graph.primitive_standardized || !primitive.supported) {
    return FailBuiltGraphValidation("unsupported_fold_operation", required_scratch_slots);
  }
  if (graph.partition_count != partition_count ||
      graph.partition_fold_slots.size() != static_cast<std::size_t>(partition_count) ||
      graph.nodes.size() != static_cast<std::size_t>(required_nodes) ||
      graph.reduction_edges.size() != static_cast<std::size_t>(required_edges) ||
      graph.scratch_slot_count != required_scratch_slots) {
    return FailBuiltGraphValidation("fold_graph_partition_mismatch", required_scratch_slots);
  }
  return FoldGraphValidationResult{
      .ok = true,
      .reason = "pass",
      .scratch_slot_count = required_scratch_slots,
  };
}

} // namespace rund::kernel::reduction::graph::validate_detail
