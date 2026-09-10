#include "local.hpp"

namespace rund::kernel::fusion_detail {

BoundaryPlan RejectBoundary(const char *const reason, const u64 decision,
                            const u64 intermediate) noexcept {
  return BoundaryPlan{.rejected_edges = 1u,
                      .intermediate = intermediate,
                      .decision = decision,
                      .reason = reason};
}

BoundaryPlan EvaluateBoundary(const Graph &graph, const FusionPolicy &policy,
                              const BoundaryShape &shape,
                              const ReaderFact *const facts,
                              const u64 fact_count,
                              const u64 left_index) noexcept {
  const GraphNode &left = graph.nodes[left_index];
  const GraphNode &right = graph.nodes[left_index + 1u];
  if (!shape.candidate) {
    return {};
  }
  if (policy.nodes[left_index].writes_visible) {
    return RejectBoundary("compute_fusion_visibility_boundary",
                          kBoundaryDecisionVisibilityBoundary,
                          shape.intermediate);
  }
  if (shape.producer_writes != 1u || shape.consumer_reads != 1u ||
      ReaderCount(facts, fact_count, shape.intermediate, left_index) != 1u) {
    return RejectBoundary("compute_fusion_dependency_conflict",
                          kBoundaryDecisionDependencyConflict,
                          shape.intermediate);
  }
  if (left.kind != NodeKind::Map || right.kind != NodeKind::Map ||
      !policy.nodes[left_index].supported ||
      !policy.nodes[left_index + 1u].supported ||
      left.element_count != right.element_count) {
    return RejectBoundary("compute_fusion_unsupported_op",
                          kBoundaryDecisionUnsupportedOp, shape.intermediate);
  }
  if (shape.consumer_read_ordinal >= 64u ||
      (policy.nodes[left_index + 1u].direct_read_mask &
       (u64{1u} << shape.consumer_read_ordinal)) == 0u) {
    return RejectBoundary("compute_fusion_dependency_conflict",
                          kBoundaryDecisionDependencyConflict,
                          shape.intermediate);
  }
  return BoundaryPlan{.intermediate = shape.intermediate,
                      .decision = kBoundaryDecisionFused,
                      .fused = true};
}

bool ValidPolicy(const FusionPolicy &policy, const u64 node_count) noexcept {
  if (policy.node_count != node_count ||
      policy.node_count > kMaxFusionPolicyNodeCount ||
      (policy.node_count != 0u && policy.nodes == nullptr)) {
    return false;
  }
  for (u64 index = 0u; index < policy.node_count; ++index) {
    const FusionNodePolicy &node = policy.nodes[index];
    if (node.supported !=
        (node.binding_count != 0u && node.ir_node_count != 0u)) {
      return false;
    }
    if (node.binding_count > kMaxComputeBindingCount ||
        node.ir_node_count > kMaxComputeNodeCount) {
      return false;
    }
  }
  return true;
}

bool MergeFits(const u64 binding_count, const u64 ir_node_count,
               const FusionNodePolicy &right, u64 &merged_bindings,
               u64 &merged_ir_nodes) noexcept {
  merged_bindings = binding_count + right.binding_count - 2u;
  merged_ir_nodes = ir_node_count + right.ir_node_count - 2u;
  return merged_bindings != 0u && merged_ir_nodes != 0u &&
         merged_bindings <= kMaxComputeBindingCount &&
         merged_ir_nodes <= kMaxComputeNodeCount;
}

} // namespace rund::kernel::fusion_detail
