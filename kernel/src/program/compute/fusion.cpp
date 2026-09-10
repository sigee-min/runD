#include "fusion/local.hpp"

#include <new>
#include <vector>

namespace rund::kernel {

FusionPlan PlanFusion(const Graph &graph, const FusionPolicy &policy) noexcept {
  using namespace fusion_detail;
  const GraphCheck input = ValidateGraph(graph);
  if (!input.ok) {
    return FusionPlan{.original_node_count = graph.node_count,
                      .reason = input.reason};
  }
  if (!ValidPolicy(policy, graph.node_count)) {
    return FusionPlan{.input_graph_id_hi = input.graph_id_hi,
                      .input_graph_id_lo = input.graph_id_lo,
                      .original_node_count = graph.node_count,
                      .fused_node_count = graph.node_count,
                      .reason = "compute_fusion_policy_invalid"};
  }

  std::vector<BoundaryShape> shapes;
  std::vector<ReaderFact> facts;
  try {
    shapes.resize(static_cast<std::size_t>(graph.node_count));
    facts.resize(static_cast<std::size_t>(graph.node_count));
  } catch (const std::bad_alloc &) {
    return FusionPlan{.input_graph_id_hi = input.graph_id_hi,
                      .input_graph_id_lo = input.graph_id_lo,
                      .original_node_count = graph.node_count,
                      .fused_node_count = graph.node_count,
                      .reason = "compute_fusion_capacity"};
  }
  const u64 fact_count = BuildReaderFacts(graph, shapes, facts);
  FusionPlan result{};
  u64 fused_boundary_count = 0u;
  u64 rejected_edge_count = 0u;
  const char *first_rejection = "compute_fusion_ok";
  FusionHash decisions{.hi = 0xc6a4a7935bd1e995ull,
                       .lo = 0x9e3779b97f4a7c15ull};
  u64 region_bindings = policy.nodes[0].binding_count;
  u64 region_ir_nodes = policy.nodes[0].ir_node_count;
  for (u64 boundary_index = 0u; boundary_index + 1u < graph.node_count;
       ++boundary_index) {
    BoundaryPlan boundary =
        EvaluateBoundary(graph, policy, shapes[boundary_index], facts.data(),
                         fact_count, boundary_index);
    if (boundary.fused) {
      u64 merged_bindings = 0u;
      u64 merged_ir_nodes = 0u;
      if (MergeFits(region_bindings, region_ir_nodes,
                    policy.nodes[boundary_index + 1u], merged_bindings,
                    merged_ir_nodes)) {
        region_bindings = merged_bindings;
        region_ir_nodes = merged_ir_nodes;
      } else {
        boundary = RejectBoundary("compute_fusion_capacity_boundary",
                                  kBoundaryDecisionCapacityBoundary,
                                  boundary.intermediate);
      }
    }
    if (!boundary.fused) {
      region_bindings = policy.nodes[boundary_index + 1u].binding_count;
      region_ir_nodes = policy.nodes[boundary_index + 1u].ir_node_count;
    }
    decisions = Mix(decisions, boundary_index);
    decisions = Mix(decisions, boundary.decision);
    decisions = Mix(decisions, boundary.intermediate);
    decisions = Mix(decisions, boundary.rejected_edges);
    decisions = Mix(decisions, boundary.fused ? 1u : 0u);
    if (boundary.rejected_edges != 0u && rejected_edge_count == 0u) {
      first_rejection = boundary.reason;
    }
    rejected_edge_count += boundary.rejected_edges;
    if (boundary.fused) {
      ++fused_boundary_count;
      result.fused_boundaries[boundary_index / 64u] |=
          u64{1u} << (boundary_index % 64u);
    }
  }

  const u64 fused_node_count = graph.node_count - fused_boundary_count;
  const FusionHash output =
      FusedOutputId(input, policy, graph.node_count, fused_node_count,
                    rejected_edge_count, decisions);
  result.input_graph_id_hi = input.graph_id_hi;
  result.input_graph_id_lo = input.graph_id_lo;
  result.output_graph_id_hi = output.hi;
  result.output_graph_id_lo = output.lo;
  result.original_node_count = graph.node_count;
  result.fused_node_count = fused_node_count;
  result.rejected_edge_count = rejected_edge_count;
  result.ok = true;
  result.reason =
      rejected_edge_count == 0u ? "compute_fusion_ok" : first_rejection;
  return result;
}

} // namespace rund::kernel
