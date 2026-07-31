#include <kernel/reduction/fold/graph/api.hpp>

namespace rund::kernel {

void ResetFoldGraph(FoldGraph& graph) {
  graph.operation = FoldOperation::FixedBinaryTreeHash;
  graph.value_domain = FoldValueDomain::HashDigest;
  graph.strict_float_reduction = StrictFloatReductionPolicy{};
  graph.partition_count = 0u;
  graph.worker_local_slot_count = 0u;
  graph.global_ordered_slot_count = 0u;
  graph.scratch_slot_count = 0u;
  graph.final_slot = 0u;
  graph.fixed_binary_tree = true;
  graph.fixed_topological_order = true;
  graph.dag_validated = false;
  graph.slot_bounds_validated = false;
  graph.padding_law_validated = false;
  graph.primitive_standardized = true;
  graph.floating_point_allowed = false;
  graph.strict_floating_point = false;
  graph.partition_fold_slots.clear();
  graph.nodes.clear();
  graph.reduction_edges.clear();
}

bool ReserveFoldGraph(FoldGraph& graph, const u32 partition_capacity) {
  try {
    graph.partition_fold_slots.reserve(partition_capacity);
    graph.nodes.reserve(FoldGraphFixedBinaryTreeNodeCount(partition_capacity));
    graph.reduction_edges.reserve(FoldGraphFixedBinaryTreeEdgeCount(partition_capacity));
  } catch (...) {
    return false;
  }
  return true;
}

} // namespace rund::kernel
