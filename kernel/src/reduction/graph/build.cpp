#include "build/local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

namespace rund::kernel {

FoldGraphBuild BuildFoldGraph(FoldGraph& graph,
                              const u32 partition_count,
                              const FoldOperation operation,
                              const AllocationPolicy allocation) {
  return BuildFoldGraph(graph, partition_count, operation, StrictFloatReductionPolicy{}, allocation);
}

FoldGraphBuild BuildFoldGraph(FoldGraph& graph,
                              const u32 partition_count,
                              const FoldOperation operation,
                              const StrictFloatReductionPolicy strict_float_reduction,
                              const AllocationPolicy allocation) {
  const FoldPrimitiveSpec primitive = DescribeFoldOperation(operation);
  const reduction::graph::build_detail::BuildPlan plan{
      .partition_count = partition_count,
      .required_edges = FoldGraphFixedBinaryTreeEdgeCount(partition_count),
      .required_nodes = FoldGraphFixedBinaryTreeNodeCount(partition_count),
      .required_scratch_slots = FoldGraphFixedBinaryTreeScratchSlotCount(partition_count),
      .operation = operation,
      .primitive = primitive,
      .strict_float_reduction = strict_float_reduction,
      .allocation = allocation,
  };
  if (!IsSupportedFoldOperation(operation)) {
    return reduction::graph::build_detail::PrimitiveFailure(plan, "unsupported_fold_operation");
  }
  if (primitive.admits_floating_point && !strict_float_reduction.valid) {
    return reduction::graph::build_detail::PrimitiveFailure(plan, "floating_point_fold_forbidden");
  }
  if (primitive.admits_floating_point && !StrictFloatReductionValid(operation, strict_float_reduction)) {
    return reduction::graph::build_detail::PrimitiveFailure(plan, "floating_point_law_mismatch");
  }
  if (partition_count == 0u) {
    return reduction::graph::build_detail::PrimitiveFailure(plan, "invalid_partition_count");
  }

  if (plan.required_nodes == 0u || plan.required_scratch_slots == 0u) {
    return reduction::graph::build_detail::CapacityFailure(plan, "fold_graph_capacity_exceeded");
  }

  if (allocation == AllocationPolicy::NoGrowth &&
      (graph.partition_fold_slots.capacity() < partition_count ||
       graph.nodes.capacity() < plan.required_nodes ||
       graph.reduction_edges.capacity() < plan.required_edges)) {
    return reduction::graph::build_detail::CapacityFailure(plan, "fold_graph_capacity_exceeded");
  }

  if (allocation == AllocationPolicy::AllowGrowth) {
    graph.partition_fold_slots.reserve(partition_count);
    graph.nodes.reserve(plan.required_nodes);
    graph.reduction_edges.reserve(plan.required_edges);
  }
  reduction::graph::build_detail::PopulateFixedBinaryTreeGraph(graph, plan);
  const FoldGraphValidationResult validation =
      reduction::graph::ValidateBuiltFixedBinaryTreeGraph(graph,
                                                                 operation,
                                                                 primitive,
                                                                 partition_count,
                                                                 plan.required_nodes,
                                                                 plan.required_edges,
                                                                 plan.required_scratch_slots);
  if (!validation.ok) {
    graph.dag_validated = false;
    graph.slot_bounds_validated = false;
    graph.padding_law_validated = false;
    return reduction::graph::build_detail::ValidationFailure(graph, plan, validation);
  }
  graph.scratch_slot_count = validation.scratch_slot_count;
  graph.final_slot = validation.final_slot;
  graph.dag_validated = validation.dag_validated;
  graph.slot_bounds_validated = validation.slot_bounds_validated;
  graph.padding_law_validated = validation.padding_law_validated;

  return reduction::graph::build_detail::Success(graph, plan, validation);
}

} // namespace rund::kernel
