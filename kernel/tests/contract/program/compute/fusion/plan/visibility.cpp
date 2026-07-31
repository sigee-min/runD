#include "contract/program/compute/fusion/plan/local.hpp"

namespace program_compute_contract {
namespace {

using namespace fusion_support;

int test_compute_fusion_multi_write_cpu_visible_boundary_rejects() {
  const MultiWriteProducerGraphFixture fixture{};
  const rund::kernel::FusionNodePolicy nodes[] = {
      SupportedNode(true),
      SupportedNode(),
  };
  const rund::kernel::FusionPolicy policy = SupportedPolicy(nodes, 2u);

  const auto plan = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} ==
              "compute_fusion_visibility_boundary");
  TEST_ASSERT(plan.original_node_count == 2u);
  TEST_ASSERT(plan.fused_node_count == 2u);
  TEST_ASSERT(plan.rejected_edge_count == 1u);
  TEST_ASSERT(!SameId(plan.input_graph_id_hi, plan.input_graph_id_lo,
                      plan.output_graph_id_hi, plan.output_graph_id_lo));
  return 0;
}

int test_compute_fusion_mixed_plan_preserves_first_rejection_reason() {
  const ThreeNodeChainGraphFixture fixture{};
  const rund::kernel::FusionNodePolicy nodes[] = {
      SupportedNode(true),
      SupportedNode(),
      SupportedNode(),
  };
  const rund::kernel::FusionPolicy policy = SupportedPolicy(nodes, 3u);

  const auto plan = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} ==
              "compute_fusion_visibility_boundary");
  TEST_ASSERT(plan.original_node_count == 3u);
  TEST_ASSERT(plan.fused_node_count == 2u);
  TEST_ASSERT(plan.rejected_edge_count == 1u);
  TEST_ASSERT(!SameId(plan.input_graph_id_hi, plan.input_graph_id_lo,
                      plan.output_graph_id_hi, plan.output_graph_id_lo));
  return 0;
}

int test_compute_fusion_output_id_includes_boundary_decisions() {
  const ThreeNodeChainGraphFixture fixture{};
  const rund::kernel::FusionNodePolicy first_nodes[] = {
      SupportedNode(true),
      SupportedNode(),
      SupportedNode(),
  };
  const rund::kernel::FusionNodePolicy second_nodes[] = {
      SupportedNode(),
      SupportedNode(true),
      SupportedNode(),
  };
  const rund::kernel::FusionPolicy first_policy =
      SupportedPolicy(first_nodes, 3u);
  const rund::kernel::FusionPolicy second_policy =
      SupportedPolicy(second_nodes, 3u);

  const auto first =
      rund::kernel::PlanFusion(fixture.graph, first_policy);
  const auto second =
      rund::kernel::PlanFusion(fixture.graph, second_policy);

  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(first.original_node_count == second.original_node_count);
  TEST_ASSERT(first.fused_node_count == second.fused_node_count);
  TEST_ASSERT(first.rejected_edge_count == second.rejected_edge_count);
  TEST_ASSERT(!SameId(first.output_graph_id_hi, first.output_graph_id_lo,
                      second.output_graph_id_hi, second.output_graph_id_lo));
  return 0;
}

int test_compute_fusion_cpu_visible_boundary_keeps_unfused_plan() {
  const LinearGraphFixture fixture{};
  const rund::kernel::FusionNodePolicy nodes[] = {
      SupportedNode(true),
      SupportedNode(),
  };
  const rund::kernel::FusionPolicy policy = SupportedPolicy(nodes, 2u);

  const auto plan = rund::kernel::PlanFusion(fixture.graph, policy);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} ==
              "compute_fusion_visibility_boundary");
  TEST_ASSERT(plan.original_node_count == 2u);
  TEST_ASSERT(plan.fused_node_count == 2u);
  TEST_ASSERT(plan.rejected_edge_count == 1u);
  TEST_ASSERT(!SameId(plan.input_graph_id_hi, plan.input_graph_id_lo,
                      plan.output_graph_id_hi, plan.output_graph_id_lo));
  return 0;
}

int test_compute_fusion_unfused_output_id_includes_rejection_decision() {
  const LinearGraphFixture fixture{};
  const rund::kernel::FusionNodePolicy visible_nodes[] = {
      SupportedNode(true),
      SupportedNode(),
  };
  const rund::kernel::FusionNodePolicy unsupported_nodes[] = {
      SupportedNode(),
      {.supported = false},
  };
  const rund::kernel::FusionPolicy visible_policy =
      SupportedPolicy(visible_nodes, 2u);
  const rund::kernel::FusionPolicy unsupported_policy =
      SupportedPolicy(unsupported_nodes, 2u);

  const auto visible =
      rund::kernel::PlanFusion(fixture.graph, visible_policy);
  const auto unsupported =
      rund::kernel::PlanFusion(fixture.graph, unsupported_policy);

  TEST_ASSERT(visible.ok);
  TEST_ASSERT(unsupported.ok);
  TEST_ASSERT(visible.original_node_count == unsupported.original_node_count);
  TEST_ASSERT(visible.fused_node_count == unsupported.fused_node_count);
  TEST_ASSERT(visible.rejected_edge_count == unsupported.rejected_edge_count);
  TEST_ASSERT(std::string_view{visible.reason} ==
              "compute_fusion_visibility_boundary");
  TEST_ASSERT(std::string_view{unsupported.reason} ==
              "compute_fusion_unsupported_op");
  TEST_ASSERT(!SameId(visible.output_graph_id_hi, visible.output_graph_id_lo,
                      unsupported.output_graph_id_hi,
                      unsupported.output_graph_id_lo));
  return 0;
}

} // namespace

int RunFusionPlanVisibilityContract() {
  if (test_compute_fusion_multi_write_cpu_visible_boundary_rejects() != 0) {
    return 1;
  }
  if (test_compute_fusion_mixed_plan_preserves_first_rejection_reason() != 0) {
    return 1;
  }
  if (test_compute_fusion_output_id_includes_boundary_decisions() != 0) {
    return 1;
  }
  if (test_compute_fusion_cpu_visible_boundary_keeps_unfused_plan() != 0) {
    return 1;
  }
  return test_compute_fusion_unfused_output_id_includes_rejection_decision();
}

} // namespace program_compute_contract
