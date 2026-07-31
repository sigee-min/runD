#include "local.hpp"

namespace rund::kernel::reduction::graph::validate_detail {

FoldGraphValidationResult ValidateFixedGraphPartitionNodes(const FoldGraph& graph,
                                                           const FoldOperation operation,
                                                           const FoldPrimitiveSpec& primitive,
                                                           const u32 partition_count,
                                                           const u32 required_scratch_slots) {
  for (u32 partition = 0u; partition < partition_count; ++partition) {
    if (graph.partition_fold_slots[partition] != partition) {
      return FailBuiltGraphValidation("fold_graph_slot_out_of_range", required_scratch_slots);
    }
    const FoldGraphNode& worker_node = graph.nodes[partition * 2u];
    const FoldGraphNode& ordered_node = graph.nodes[partition * 2u + 1u];
    if (worker_node.kind != FoldGraphNodeKind::WorkerLocalPartial ||
        ordered_node.kind != FoldGraphNodeKind::GlobalOrderedSlot ||
        worker_node.topological_index != partition * 2u ||
        ordered_node.topological_index != partition * 2u + 1u ||
        worker_node.slot != partition ||
        ordered_node.slot != partition ||
        worker_node.operation != operation ||
        ordered_node.operation != operation ||
        worker_node.value_domain != primitive.value_domain ||
        ordered_node.value_domain != primitive.value_domain ||
        worker_node.overflow_law != primitive.overflow_law ||
        ordered_node.overflow_law != primitive.overflow_law) {
      return FailBuiltGraphValidation("fold_graph_node_mismatch", required_scratch_slots);
    }
  }
  return FoldGraphValidationResult{
      .ok = true,
      .reason = "pass",
      .scratch_slot_count = required_scratch_slots,
  };
}

} // namespace rund::kernel::reduction::graph::validate_detail
