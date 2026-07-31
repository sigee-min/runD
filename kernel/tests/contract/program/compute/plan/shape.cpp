#include "local.hpp"
#include "test/compute/fixed.hpp"

namespace program_compute_contract {

int test_compute_plan_computes_dispatch_windows() {
  const rund::kernel::ComputePlan plan =
      rund::kernel::PlanCompute(ComputePhase(), ComputeFixedMap(),
                                ComputeMetalCaps(), ComputeDispatchLimit());

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(rund::kernel::ComputePlanShapeValid(plan));
  TEST_ASSERT(std::string_view{plan.reason} == "ok");
  TEST_ASSERT(plan.phase_id == 91u);
  TEST_ASSERT(plan.tile_count == 10u);
  TEST_ASSERT(plan.op_hash_hi == 0x1020304050607080u);
  TEST_ASSERT(plan.op_hash_lo == 0x8877665544332211u);
  TEST_ASSERT(plan.api == rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(plan.scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(plan.fixed_format == ComputeFixedMap().fixed_format);
  TEST_ASSERT(plan.input_buffer_count == 2u);
  TEST_ASSERT(plan.input_bytes_per_tile == 32u);
  TEST_ASSERT(plan.output_bytes_per_tile == 16u);
  TEST_ASSERT(plan.param_bytes == 8u);
  TEST_ASSERT(plan.metadata_bytes_per_tile == 8u);
  TEST_ASSERT(plan.bytes_per_tile == 56u);
  TEST_ASSERT(plan.staging_bytes == 232u);
  TEST_ASSERT(plan.dispatch_window_tiles == 4u);
  TEST_ASSERT(plan.dispatch_count == 3u);
  TEST_ASSERT(plan.fixed_authoritative);
  return 0;
}

int test_compute_plan_shape_guard_rejects_forgery() {
  const rund::kernel::ComputeMap map = ComputeFixedMap();
  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), map, ComputeMetalCaps(), ComputeDispatchLimit());

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(rund::kernel::ComputePlanShapeValid(plan));
  TEST_ASSERT(rund::kernel::ComputePlanBytesValid(plan));
  TEST_ASSERT(rund::kernel::ComputePlanMatchesMap(plan, map));

  rund::kernel::ComputePlan forged_count = plan;
  forged_count.dispatch_count = 2u;
  TEST_ASSERT(!rund::kernel::ComputePlanShapeValid(forged_count));

  rund::kernel::ComputePlan forged_staging = plan;
  forged_staging.staging_bytes += 1u;
  TEST_ASSERT(!rund::kernel::ComputePlanBytesValid(forged_staging));
  TEST_ASSERT(!rund::kernel::ComputePlanShapeValid(forged_staging));

  rund::kernel::ComputePlan forged_output = plan;
  forged_output.output_bytes_per_tile = sizeof(rund::kernel::u32);
  TEST_ASSERT(!rund::kernel::ComputePlanMatchesMap(forged_output, map));

  rund::kernel::ComputePlan forged_format = plan;
  forged_format.fixed_format.fraction_bits = 63u;
  TEST_ASSERT(!rund::kernel::ComputePlanMatchesMap(forged_format, map));

  rund::kernel::ComputeMap fixed_lane64 = map;
  fixed_lane64.scalar = rund::kernel::ComputeScalar::Lane64;
  fixed_lane64.fixed_format =
      test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane64);
  const rund::kernel::ComputePlan plan64 = rund::kernel::PlanCompute(
      ComputePhase(), fixed_lane64, ComputeMetalCaps(), ComputeDispatchLimit());
  TEST_ASSERT(plan64.ok);
  TEST_ASSERT(plan64.fixed_format == fixed_lane64.fixed_format);
  TEST_ASSERT(rund::kernel::ComputePlanMatchesMap(plan64, fixed_lane64));

  return 0;
}

int test_compute_plan_scalar_guard_is_separate() {
  const rund::kernel::ComputePlan generic =
      rund::kernel::PlanCompute(ComputePhase(), ComputeFixedMap(),
                                ComputeMetalCaps(), ComputeDispatchLimit());
  TEST_ASSERT(generic.ok);
  TEST_ASSERT(rund::kernel::ComputePlanShapeValid(generic));
  TEST_ASSERT(!rund::kernel::ComputePlanScalarValid(generic));

  rund::kernel::ComputeMap scalar_tight = ComputeFixedMap();
  scalar_tight.input_bytes_per_tile = 8u;
  scalar_tight.output_bytes_per_tile = 4u;
  scalar_tight.metadata_bytes_per_tile = 0u;
  const rund::kernel::ComputePlan backend_abi = rund::kernel::PlanCompute(
      ComputePhase(), scalar_tight, ComputeMetalCaps(), ComputeDispatchLimit());
  TEST_ASSERT(backend_abi.ok);
  TEST_ASSERT(rund::kernel::ComputePlanShapeValid(backend_abi));
  TEST_ASSERT(rund::kernel::ComputePlanScalarValid(backend_abi));

  rund::kernel::ComputeMap widened_mask = scalar_tight;
  widened_mask.input_buffer_count = 1u;
  widened_mask.input_bytes_per_tile = sizeof(rund::kernel::u32);
  widened_mask.output_bytes_per_tile = sizeof(rund::kernel::u64);
  const rund::kernel::ComputePlan widened_mask_plan = rund::kernel::PlanCompute(
      ComputePhase(), widened_mask, ComputeMetalCaps(), ComputeDispatchLimit());
  TEST_ASSERT(widened_mask_plan.ok);
  TEST_ASSERT(rund::kernel::ComputePlanShapeValid(widened_mask_plan));
  TEST_ASSERT(rund::kernel::ComputePlanScalarValid(widened_mask_plan));

  rund::kernel::ComputePlan widened_multi_output = widened_mask_plan;
  widened_multi_output.output_buffer_count = 2u;
  widened_multi_output.output_bytes_per_tile = 2u * sizeof(rund::kernel::u64);
  TEST_ASSERT(!rund::kernel::ComputePlanScalarValid(widened_multi_output));

  rund::kernel::ComputePlan widened_input_mismatch = widened_mask_plan;
  widened_input_mismatch.input_bytes_per_tile = sizeof(rund::kernel::u16);
  TEST_ASSERT(!rund::kernel::ComputePlanScalarValid(widened_input_mismatch));

  rund::kernel::ComputePlan indexed_lane64 = backend_abi;
  indexed_lane64.scalar = rund::kernel::ComputeScalar::Lane64;
  indexed_lane64.input_buffer_count = 2u;
  indexed_lane64.input_bytes_per_tile =
      sizeof(rund::kernel::u64) + sizeof(rund::kernel::u32);
  indexed_lane64.output_bytes_per_tile = sizeof(rund::kernel::u64);
  indexed_lane64.param_bytes = sizeof(rund::kernel::u64);
  TEST_ASSERT(rund::kernel::ComputePlanScalarValid(indexed_lane64));

  indexed_lane64.input_bytes_per_tile += 1u;
  TEST_ASSERT(!rund::kernel::ComputePlanScalarValid(indexed_lane64));
  return 0;
}

} // namespace program_compute_contract
