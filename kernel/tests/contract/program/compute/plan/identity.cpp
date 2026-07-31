#include "local.hpp"

namespace program_compute_contract {

int test_compute_plan_is_deterministic_for_same_inputs() {
  const rund::kernel::ComputePlan first =
      rund::kernel::PlanCompute(ComputePhase(), ComputeFixedMap(),
                                ComputeMetalCaps(), ComputeDispatchLimit());
  const rund::kernel::ComputePlan second =
      rund::kernel::PlanCompute(ComputePhase(), ComputeFixedMap(),
                                ComputeMetalCaps(), ComputeDispatchLimit());

  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(first.phase_id == second.phase_id);
  TEST_ASSERT(first.tile_count == second.tile_count);
  TEST_ASSERT(first.op_hash_hi == second.op_hash_hi);
  TEST_ASSERT(first.op_hash_lo == second.op_hash_lo);
  TEST_ASSERT(first.api == second.api);
  TEST_ASSERT(first.scalar == second.scalar);
  TEST_ASSERT(first.input_buffer_count == second.input_buffer_count);
  TEST_ASSERT(first.input_bytes_per_tile == second.input_bytes_per_tile);
  TEST_ASSERT(first.output_bytes_per_tile == second.output_bytes_per_tile);
  TEST_ASSERT(first.param_bytes == second.param_bytes);
  TEST_ASSERT(first.metadata_bytes_per_tile == second.metadata_bytes_per_tile);
  TEST_ASSERT(first.bytes_per_tile == second.bytes_per_tile);
  TEST_ASSERT(first.staging_bytes == second.staging_bytes);
  TEST_ASSERT(first.dispatch_window_tiles == second.dispatch_window_tiles);
  TEST_ASSERT(first.dispatch_count == second.dispatch_count);
  TEST_ASSERT(first.fixed_authoritative == second.fixed_authoritative);
  TEST_ASSERT(std::string_view{first.reason} ==
              std::string_view{second.reason});
  return 0;
}

} // namespace program_compute_contract
