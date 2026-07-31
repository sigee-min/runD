#pragma once

#include <kernel/program/compute/plan.hpp>

#include "test/compute/fixed.hpp"

#include "local.hpp"

namespace node_accel_contract::policy_case {

[[nodiscard]] inline rund::kernel::ComputePlan
RuntimePolicyPlanFor(const rund::kernel::ComputeApi api) {
  const rund::kernel::TilePhaseDescription phase{
      .phase_id = 77u,
      .tile_count = 8u,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{.scratch_bytes_per_tile =
                                                         0u,
                                                     .scratch_alignment = 1u,
                                                     .output_shards = 8u,
                                                     .queue_slots = 8u,
                                                     .task_slots = 8u},
  };
  const rund::kernel::ComputeMap map{
      .op_hash_hi = 0x1234u,
      .op_hash_lo = 0x5678u,
      .api = api,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = rund::kernel::ComputeDomain::Fixed,
      .fixed_format =
          test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32),
      .input_buffer_count = 1u,
      .input_bytes_per_tile = 4u,
      .output_bytes_per_tile = 4u,
      .param_bytes = 4u,
      .metadata_bytes_per_tile = 0u,
  };
  const rund::kernel::ComputeCaps caps{
      .api = api,
      .device_bytes = 4096u,
      .staging_bytes = 4096u,
      .max_window_tiles = 4u,
      .subgroup_width = 1u,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::ComputeLimit limit{
      .staging_bytes = 4096u,
      .max_window_tiles = 4u,
  };
  return rund::kernel::PlanCompute(phase, map, caps, limit);
}

[[nodiscard]] inline rund::kernel::ComputePlan RuntimePolicyPlan() {
  return RuntimePolicyPlanFor(rund::kernel::ComputeApi::Metal);
}

} // namespace node_accel_contract::policy_case
