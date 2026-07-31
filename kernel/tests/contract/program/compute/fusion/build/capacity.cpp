#include "model.hpp"

namespace program_compute_contract::fusion_build_contract {
namespace {

int test_compute_fusion_plans_capacity_before_lowering() {
  const OversizedFusedBindingFixture fixture{};
  const rund::kernel::FusionPlan plan =
      rund::kernel::PlanFusion(fixture.graph, fixture.policy);
  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.fused_node_count == 2u);
  TEST_ASSERT(plan.rejected_edge_count == 1u);
  TEST_ASSERT(std::string_view{plan.reason} ==
              "compute_fusion_capacity_boundary");
  TEST_ASSERT(!plan.boundary_fused(0u));
  const rund::kernel::ComputeIR chain[2] = {fixture.first, fixture.second};

  const auto admitted = rund::kernel::compute_lowering_detail::
      BuildAdmittedFusedComputeMapChainIR(chain, 2u, fixture.graph,
                                          fixture.policy,
                                          rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!admitted.value.ok);
  TEST_ASSERT(std::string_view{admitted.value.reason} ==
              "compute_fusion_dependency_conflict");
  TEST_ASSERT(admitted.source_parse_count == 0u);
  TEST_ASSERT(!admitted.input.ok);
  TEST_ASSERT(admitted.input.parse_count == 0u);

  const rund::kernel::ComputeFusedMapChainIR fused =
      rund::kernel::BuildFusedComputeMapChainIR(
          chain, 2u, fixture.graph, fixture.policy,
          rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(!fused.ok);
  TEST_ASSERT(std::string_view{fused.reason} ==
              "compute_fusion_dependency_conflict");
  return 0;
}

} // namespace

int RunCapacity() {
  return test_compute_fusion_plans_capacity_before_lowering();
}

} // namespace program_compute_contract::fusion_build_contract
