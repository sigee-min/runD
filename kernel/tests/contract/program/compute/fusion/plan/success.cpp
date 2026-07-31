#include "contract/program/compute/fusion/plan/local.hpp"

#include <vector>

namespace program_compute_contract {
namespace {

using namespace fusion_support;

int test_compute_fusion_linear_map_chain_can_fuse() {
  const LinearGraphFixture fixture{};
  const rund::kernel::FusionNodePolicy nodes[] = {
      SupportedNode(),
      SupportedNode(),
  };
  const rund::kernel::FusionPolicy policy = SupportedPolicy(nodes, 2u);

  const auto first = rund::kernel::PlanFusion(fixture.graph, policy);
  const auto second = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(first.ok);
  TEST_ASSERT(std::string_view{first.reason} == "compute_fusion_ok");
  TEST_ASSERT(first.original_node_count == 2u);
  TEST_ASSERT(first.fused_node_count == 1u);
  TEST_ASSERT(first.rejected_edge_count == 0u);
  TEST_ASSERT(first.boundary_fused(0u));
  TEST_ASSERT(first.input_graph_id_hi != 0u || first.input_graph_id_lo != 0u);
  TEST_ASSERT(first.output_graph_id_hi != 0u || first.output_graph_id_lo != 0u);
  TEST_ASSERT(!SameId(first.input_graph_id_hi, first.input_graph_id_lo,
                      first.output_graph_id_hi, first.output_graph_id_lo));
  TEST_ASSERT(SameId(first.output_graph_id_hi, first.output_graph_id_lo,
                     second.output_graph_id_hi, second.output_graph_id_lo));
  return 0;
}

int test_compute_fusion_multiple_consumers_keep_unfused_plan() {
  const BranchedConsumerGraphFixture fixture{};
  const rund::kernel::FusionNodePolicy nodes[] = {
      SupportedNode(),
      SupportedNode(),
      SupportedNode(),
  };
  const rund::kernel::FusionPolicy policy = SupportedPolicy(nodes, 3u);

  const auto plan = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} ==
              "compute_fusion_dependency_conflict");
  TEST_ASSERT(plan.original_node_count == 3u);
  TEST_ASSERT(plan.fused_node_count == 3u);
  TEST_ASSERT(plan.rejected_edge_count == 1u);
  TEST_ASSERT(!plan.boundary_fused(0u));
  TEST_ASSERT(!plan.boundary_fused(1u));
  TEST_ASSERT(!SameId(plan.input_graph_id_hi, plan.input_graph_id_lo,
                      plan.output_graph_id_hi, plan.output_graph_id_lo));
  return 0;
}

int test_compute_fusion_maximum_chain_is_one_region() {
  constexpr rund::kernel::u64 count = rund::kernel::kMaxGraphNodeCount;
  static_assert(rund::kernel::kMaxGraphNodeCount >
                rund::kernel::kMaxComputeNodeCount);
  std::vector<rund::kernel::GraphBufferRef> buffers(count * 2u);
  std::vector<rund::kernel::GraphNode> graph_nodes(count);
  std::vector<rund::kernel::FusionNodePolicy> policy_nodes(count);

  for (rund::kernel::u64 index = 0u; index < count; ++index) {
    buffers[index * 2u] = rund::kernel::GraphBufferRef{
        .logical_id = index + 1u,
        .role = rund::kernel::BufferRole::Read,
    };
    buffers[index * 2u + 1u] = rund::kernel::GraphBufferRef{
        .logical_id = index + 2u,
        .role = rund::kernel::BufferRole::Write,
    };
    graph_nodes[index] = rund::kernel::GraphNode{
        .op_hash_hi = kFirstOp.op_hash_hi,
        .op_hash_lo = kFirstOp.op_hash_lo,
        .buffers = buffers.data() + index * 2u,
        .buffer_count = 2u,
        .element_count = 1024u,
    };
    policy_nodes[index] = SupportedNode();
  }

  const rund::kernel::Graph graph{
      .nodes = graph_nodes.data(),
      .node_count = count,
      .scalar = rund::kernel::ComputeScalar::Lane32,
  };
  const rund::kernel::FusionPolicy policy =
      SupportedPolicy(policy_nodes.data(), count);
  const auto first = rund::kernel::PlanFusion(graph, policy);
  const auto second = rund::kernel::PlanFusion(graph, policy);

  TEST_ASSERT(first.ok);
  TEST_ASSERT(first.original_node_count == count);
  TEST_ASSERT(first.fused_node_count == 1u);
  TEST_ASSERT(first.rejected_edge_count == 0u);
  for (rund::kernel::u64 index = 0u; index + 1u < count; ++index) {
    TEST_ASSERT(first.boundary_fused(index));
  }
  TEST_ASSERT(!first.boundary_fused(count - 1u));
  TEST_ASSERT(!first.boundary_fused(~rund::kernel::u64{0u}));
  TEST_ASSERT(SameId(first.output_graph_id_hi, first.output_graph_id_lo,
                     second.output_graph_id_hi, second.output_graph_id_lo));
  return 0;
}

} // namespace

int RunFusionPlanSuccessContract() {
  if (test_compute_fusion_linear_map_chain_can_fuse() != 0) {
    return 1;
  }
  if (test_compute_fusion_multiple_consumers_keep_unfused_plan() != 0) {
    return 1;
  }
  return test_compute_fusion_maximum_chain_is_one_region();
}

} // namespace program_compute_contract
