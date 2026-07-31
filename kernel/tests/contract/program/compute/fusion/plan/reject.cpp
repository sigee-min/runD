#include "contract/program/compute/fusion/plan/local.hpp"

namespace program_compute_contract {
namespace {

using namespace fusion_support;

int test_compute_fusion_unsupported_op_keeps_valid_unfused_plan() {
  const LinearGraphFixture fixture{kUnsupportedOp};
  const rund::kernel::FusionNodePolicy nodes[] = {
      SupportedNode(),
      {.supported = false},
  };
  const rund::kernel::FusionPolicy policy = SupportedPolicy(nodes, 2u);

  const auto plan = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_fusion_unsupported_op");
  TEST_ASSERT(plan.original_node_count == 2u);
  TEST_ASSERT(plan.fused_node_count == 2u);
  TEST_ASSERT(plan.rejected_edge_count == 1u);
  TEST_ASSERT(!SameId(plan.input_graph_id_hi, plan.input_graph_id_lo,
                      plan.output_graph_id_hi, plan.output_graph_id_lo));
  return 0;
}

int test_compute_fusion_collective_boundary_keeps_valid_unfused_plan() {
  LinearGraphFixture fixture{};
  fixture.nodes[1] = rund::kernel::GraphNode{
      .buffers = fixture.second_buffers,
      .buffer_count = 2u,
      .kind = rund::kernel::NodeKind::Sort,
      .primitive_hash_hi = 0x1111222233334444ull,
      .primitive_hash_lo = 0x5555666677778888ull,
      .element_count = 256u,
  };
  const rund::kernel::FusionNodePolicy nodes[] = {
      SupportedNode(),
      {.supported = false},
  };
  const rund::kernel::FusionPolicy policy = SupportedPolicy(nodes, 2u);

  const auto plan = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_fusion_unsupported_op");
  TEST_ASSERT(plan.original_node_count == 2u);
  TEST_ASSERT(plan.fused_node_count == 2u);
  TEST_ASSERT(plan.rejected_edge_count == 1u);
  TEST_ASSERT(!SameId(plan.input_graph_id_hi, plan.input_graph_id_lo,
                      plan.output_graph_id_hi, plan.output_graph_id_lo));
  return 0;
}

int test_compute_fusion_rejects_different_map_element_counts() {
  LinearGraphFixture fixture{};
  fixture.nodes[1].element_count += 1u;
  const rund::kernel::FusionNodePolicy nodes[] = {
      SupportedNode(),
      SupportedNode(),
  };
  const rund::kernel::FusionPolicy policy = SupportedPolicy(nodes, 2u);

  const auto plan = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_fusion_unsupported_op");
  TEST_ASSERT(plan.original_node_count == 2u);
  TEST_ASSERT(plan.fused_node_count == 2u);
  TEST_ASSERT(plan.rejected_edge_count == 1u);
  return 0;
}

int test_compute_fusion_rejects_indirect_consumer_edge() {
  const LinearGraphFixture fixture{};
  rund::kernel::FusionNodePolicy consumer = SupportedNode();
  consumer.direct_read_mask = 0u;
  const rund::kernel::FusionNodePolicy nodes[] = {
      SupportedNode(),
      consumer,
  };
  const rund::kernel::FusionPolicy policy = SupportedPolicy(nodes, 2u);

  const auto plan = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} ==
              "compute_fusion_dependency_conflict");
  TEST_ASSERT(plan.original_node_count == 2u);
  TEST_ASSERT(plan.fused_node_count == 2u);
  TEST_ASSERT(plan.rejected_edge_count == 1u);
  return 0;
}

int test_compute_fusion_invalid_graph_fails_whole_plan() {
  const rund::kernel::Graph graph{};
  const rund::kernel::FusionPolicy policy{};

  const auto plan = rund::kernel::PlanFusion(graph, policy);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_graph_empty");
  TEST_ASSERT(plan.original_node_count == 0u);
  TEST_ASSERT(plan.fused_node_count == 0u);
  return 0;
}

int test_compute_fusion_invalid_policy_fails_whole_plan() {
  const LinearGraphFixture fixture{};
  const rund::kernel::FusionPolicy policy{
      .nodes = nullptr,
      .node_count = 2u,
  };

  const auto plan = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_fusion_policy_invalid");
  TEST_ASSERT(plan.original_node_count == 2u);
  TEST_ASSERT(plan.fused_node_count == 2u);
  return 0;
}

int test_compute_fusion_default_plan_reason_is_invalid() {
  const rund::kernel::FusionPlan plan{};

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_fusion_invalid");
  return 0;
}

} // namespace

int RunFusionPlanRejectContract() {
  if (test_compute_fusion_unsupported_op_keeps_valid_unfused_plan() != 0) {
    return 1;
  }
  if (test_compute_fusion_collective_boundary_keeps_valid_unfused_plan() != 0) {
    return 1;
  }
  if (test_compute_fusion_rejects_different_map_element_counts() != 0) {
    return 1;
  }
  if (test_compute_fusion_rejects_indirect_consumer_edge() != 0) {
    return 1;
  }
  if (test_compute_fusion_invalid_graph_fails_whole_plan() != 0) {
    return 1;
  }
  if (test_compute_fusion_invalid_policy_fails_whole_plan() != 0) {
    return 1;
  }
  return test_compute_fusion_default_plan_reason_is_invalid();
}

} // namespace program_compute_contract
