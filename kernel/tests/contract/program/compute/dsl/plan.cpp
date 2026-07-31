#include "contract/program/compute/dsl/local.hpp"
#include "test/assert.hpp"

namespace program_compute_contract {
namespace {

using namespace dsl_support;

int test_compute_dsl_map_feeds_plan_compute() {
  const auto op = BuildIntegrateOp(7);
  const rund::kernel::TilePhaseDescription phase{
      .phase_id = 92u,
      .tile_count = 4u,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{
              .output_shards = 4u,
              .queue_slots = 4u,
              .task_slots = 4u,
          },
  };
  const rund::kernel::ComputeCaps caps{
      .api = rund::kernel::ComputeApi::Metal,
      .device_bytes = 4096u,
      .staging_bytes = 256u,
      .max_window_tiles = 4u,
      .subgroup_width = 32u,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::ComputeLimit limit{
      .staging_bytes = 256u,
      .max_window_tiles = 4u,
  };

  const rund::kernel::ComputePlan plan =
      rund::kernel::PlanCompute(phase, op.map(), caps, limit);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.op_hash_hi == op.ir().op_hash_hi);
  TEST_ASSERT(plan.op_hash_lo == op.ir().op_hash_lo);
  TEST_ASSERT(plan.scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(plan.bytes_per_tile == 16u);
  TEST_ASSERT(plan.staging_bytes == 68u);
  return 0;
}

}  // namespace

int RunComputeDslPlanContract() {
  return test_compute_dsl_map_feeds_plan_compute();
}

}  // namespace program_compute_contract
