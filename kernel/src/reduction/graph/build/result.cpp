#include "local.hpp"

namespace rund::kernel::reduction::graph::build_detail {
namespace {

bool NoAllocation(const BuildPlan& plan) {
  return plan.allocation == AllocationPolicy::NoGrowth;
}

} // namespace

FoldGraphBuild PrimitiveFailure(const BuildPlan& plan, const char* const reason) {
  return FoldGraphBuild{
      .ok = false,
      .reason = reason,
      .operation = plan.operation,
      .value_domain = plan.primitive.value_domain,
      .strict_floating_point = plan.primitive.requires_strict_floating_point,
      .no_allocation = NoAllocation(plan),
  };
}

FoldGraphBuild CapacityFailure(const BuildPlan& plan, const char* const reason) {
  return FoldGraphBuild{
      .ok = false,
      .reason = reason,
      .partition_count = plan.partition_count,
      .worker_local_slot_count = plan.partition_count,
      .global_ordered_slot_count = plan.partition_count,
      .scratch_slot_count = plan.required_scratch_slots,
      .node_count = plan.required_nodes,
      .reduction_edge_count = plan.required_edges,
      .operation = plan.operation,
      .value_domain = plan.primitive.value_domain,
      .strict_floating_point = plan.primitive.requires_strict_floating_point,
      .no_allocation = NoAllocation(plan),
  };
}

FoldGraphBuild ValidationFailure(const FoldGraph& graph,
                                 const BuildPlan& plan,
                                 const FoldGraphValidationResult& validation) {
  return FoldGraphBuild{
      .ok = false,
      .reason = validation.reason,
      .partition_count = plan.partition_count,
      .worker_local_slot_count = plan.partition_count,
      .global_ordered_slot_count = plan.partition_count,
      .scratch_slot_count = plan.required_scratch_slots,
      .node_count = static_cast<u32>(graph.nodes.size()),
      .reduction_edge_count = static_cast<u32>(graph.reduction_edges.size()),
      .final_slot = graph.final_slot,
      .operation = plan.operation,
      .value_domain = plan.primitive.value_domain,
      .strict_floating_point = plan.primitive.requires_strict_floating_point,
      .no_allocation = NoAllocation(plan),
  };
}

FoldGraphBuild Success(const FoldGraph& graph,
                       const BuildPlan& plan,
                       const FoldGraphValidationResult& validation) {
  return FoldGraphBuild{
      .ok = true,
      .reason = "pass",
      .partition_count = plan.partition_count,
      .worker_local_slot_count = plan.partition_count,
      .global_ordered_slot_count = plan.partition_count,
      .scratch_slot_count = validation.scratch_slot_count,
      .node_count = static_cast<u32>(graph.nodes.size()),
      .reduction_edge_count = static_cast<u32>(graph.reduction_edges.size()),
      .final_slot = validation.final_slot,
      .operation = plan.operation,
      .value_domain = plan.primitive.value_domain,
      .dag_validated = validation.dag_validated,
      .slot_bounds_validated = validation.slot_bounds_validated,
      .padding_law_validated = validation.padding_law_validated,
      .strict_floating_point = plan.primitive.requires_strict_floating_point,
      .no_allocation = NoAllocation(plan),
  };
}

} // namespace rund::kernel::reduction::graph::build_detail
