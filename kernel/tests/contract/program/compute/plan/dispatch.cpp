#include "local.hpp"

namespace program_compute_contract {

int test_compute_dispatch_plan_matches_full_plan() {
  const rund::kernel::ComputePlan plan =
      rund::kernel::PlanCompute(ComputePhase(), ComputeFixedMap(),
                                ComputeMetalCaps(), ComputeDispatchLimit());
  const rund::kernel::ComputeDispatchPlan dispatch =
      rund::kernel::PlanComputeDispatch(ComputePhase(), ComputeFixedMap(),
                                        ComputeMetalCaps(),
                                        ComputeDispatchLimit());

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(dispatch.ok);
  TEST_ASSERT(dispatch.bytes_per_tile == plan.bytes_per_tile);
  TEST_ASSERT(dispatch.staging_bytes == plan.staging_bytes);
  TEST_ASSERT(dispatch.dispatch_window_tiles == plan.dispatch_window_tiles);
  TEST_ASSERT(dispatch.dispatch_count == plan.dispatch_count);
  TEST_ASSERT(std::string_view{dispatch.reason} == "ok");
  return 0;
}

int test_compute_dispatch_plan_rejects_like_full_plan() {
  rund::kernel::ComputeLimit limit = ComputeDispatchLimit();
  limit.staging_bytes = 63u;

  const rund::kernel::ComputeDispatchPlan dispatch =
      rund::kernel::PlanComputeDispatch(ComputePhase(), ComputeFixedMap(),
                                        ComputeMetalCaps(), limit);

  TEST_ASSERT(!dispatch.ok);
  TEST_ASSERT(std::string_view{dispatch.reason} ==
              "compute_staging_insufficient");
  return 0;
}

} // namespace program_compute_contract
