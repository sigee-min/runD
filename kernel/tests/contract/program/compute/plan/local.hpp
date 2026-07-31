#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/plan.hpp>

#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] inline rund::kernel::TilePhaseDescription
ComputePhase() noexcept {
  return rund::kernel::TilePhaseDescription{
      .phase_id = 91u,
      .tile_count = 10u,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{
              .output_shards = 10u,
              .queue_slots = 10u,
              .task_slots = 4u,
          },
  };
}

[[nodiscard]] inline rund::kernel::ComputeMap ComputeFixedMap() noexcept {
  return rund::kernel::ComputeMap{
      .op_hash_hi = 0x1020304050607080u,
      .op_hash_lo = 0x8877665544332211u,
      .api = rund::kernel::ComputeApi::Metal,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .input_buffer_count = 2u,
      .input_bytes_per_tile = 32u,
      .output_bytes_per_tile = 16u,
      .param_bytes = 8u,
      .metadata_bytes_per_tile = 8u,
  };
}

[[nodiscard]] inline rund::kernel::ComputeCaps ComputeMetalCaps() noexcept {
  return rund::kernel::ComputeCaps{
      .api = rund::kernel::ComputeApi::Metal,
      .device_bytes = 4096u,
      .staging_bytes = 1024u,
      .max_window_tiles = 8u,
      .subgroup_width = 32u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline rund::kernel::ComputeLimit
ComputeDispatchLimit() noexcept {
  return rund::kernel::ComputeLimit{
      .staging_bytes = 256u,
      .max_window_tiles = 4u,
  };
}

int test_compute_plan_rejects_invalid_phase();
int test_compute_plan_rejects_zero_workset();
int test_compute_plan_rejects_missing_op_hash();
int test_compute_plan_rejects_overflow_workset();
int test_compute_plan_rejects_staging_insufficient();
int test_compute_plan_chunks_to_staging_capacity();
int test_compute_plan_rejects_zero_dispatch_limit();
int test_compute_plan_rejects_missing_caps();
int test_compute_plan_rejects_nonfixed_scalar();
int test_compute_plan_rejects_backend_mismatch();
int test_compute_plan_rejects_unknown_api_even_when_caps_match();
int test_compute_plan_is_deterministic_for_same_inputs();
int test_compute_plan_computes_dispatch_windows();
int test_compute_plan_shape_guard_rejects_forgery();
int test_compute_plan_scalar_guard_is_separate();
int test_compute_dispatch_plan_matches_full_plan();
int test_compute_dispatch_plan_rejects_like_full_plan();

} // namespace program_compute_contract
