#include "local.hpp"

namespace program_compute_contract {

int test_compute_plan_rejects_invalid_phase() {
  rund::kernel::TilePhaseDescription phase = ComputePhase();
  phase.phase_id = 0u;

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      phase, ComputeFixedMap(), ComputeMetalCaps(), ComputeDispatchLimit());

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_phase_invalid");
  return 0;
}

int test_compute_plan_rejects_zero_workset() {
  rund::kernel::ComputeMap map = ComputeFixedMap();
  map.input_bytes_per_tile = 0u;
  map.output_bytes_per_tile = 0u;
  map.metadata_bytes_per_tile = 0u;
  map.param_bytes = 0u;

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), map, ComputeMetalCaps(), ComputeDispatchLimit());

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_workset_zero");
  return 0;
}

int test_compute_plan_rejects_missing_op_hash() {
  rund::kernel::ComputeMap map = ComputeFixedMap();
  map.op_hash_hi = 0u;
  map.op_hash_lo = 0u;

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), map, ComputeMetalCaps(), ComputeDispatchLimit());

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_op_invalid");
  return 0;
}

int test_compute_plan_rejects_overflow_workset() {
  rund::kernel::ComputeMap map = ComputeFixedMap();
  map.input_bytes_per_tile = std::numeric_limits<rund::kernel::u64>::max();

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), map, ComputeMetalCaps(), ComputeDispatchLimit());

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_workset_overflow");
  return 0;
}

int test_compute_plan_rejects_staging_insufficient() {
  rund::kernel::ComputeLimit limit = ComputeDispatchLimit();
  limit.staging_bytes = 63u;

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), ComputeFixedMap(), ComputeMetalCaps(), limit);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_staging_insufficient");
  return 0;
}

int test_compute_plan_chunks_to_staging_capacity() {
  rund::kernel::ComputeCaps caps = ComputeMetalCaps();
  caps.max_window_tiles = 8u;
  rund::kernel::ComputeLimit limit = ComputeDispatchLimit();
  limit.staging_bytes = 176u;
  limit.max_window_tiles = 8u;

  const rund::kernel::ComputePlan plan =
      rund::kernel::PlanCompute(ComputePhase(), ComputeFixedMap(), caps, limit);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.dispatch_window_tiles == 3u);
  TEST_ASSERT(plan.dispatch_count == 4u);
  TEST_ASSERT(plan.staging_bytes == 176u);
  return 0;
}

int test_compute_plan_rejects_zero_dispatch_limit() {
  rund::kernel::ComputeLimit limit = ComputeDispatchLimit();
  limit.max_window_tiles = 0u;

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), ComputeFixedMap(), ComputeMetalCaps(), limit);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_window_zero");
  return 0;
}

int test_compute_plan_rejects_missing_caps() {
  rund::kernel::ComputeCaps caps = ComputeMetalCaps();
  caps.ok = false;
  caps.reason = "device_probe_failed";

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), ComputeFixedMap(), caps, ComputeDispatchLimit());

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_caps_invalid");
  return 0;
}

int test_compute_plan_rejects_nonfixed_scalar() {
  rund::kernel::ComputeMap map = ComputeFixedMap();
  map.scalar = static_cast<rund::kernel::ComputeScalar>(99u);

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), map, ComputeMetalCaps(), ComputeDispatchLimit());

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_scalar_unsupported");
  return 0;
}

int test_compute_plan_rejects_backend_mismatch() {
  rund::kernel::ComputeMap map = ComputeFixedMap();
  map.api = rund::kernel::ComputeApi::Vulkan;

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), map, ComputeMetalCaps(), ComputeDispatchLimit());

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_backend_mismatch");
  return 0;
}

int test_compute_plan_rejects_unknown_api_even_when_caps_match() {
  rund::kernel::ComputeMap map = ComputeFixedMap();
  map.api = static_cast<rund::kernel::ComputeApi>(0u);
  rund::kernel::ComputeCaps caps = ComputeMetalCaps();
  caps.api = static_cast<rund::kernel::ComputeApi>(0u);

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      ComputePhase(), map, caps, ComputeDispatchLimit());

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_api_unsupported");
  return 0;
}

} // namespace program_compute_contract
