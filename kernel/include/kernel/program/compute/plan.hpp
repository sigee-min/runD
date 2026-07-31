#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/model.hpp>
#include <kernel/program/phase.hpp>

namespace rund::kernel {

namespace compute_detail {

[[nodiscard]] constexpr u64 Min4(const u64 a, const u64 b, const u64 c,
                                 const u64 d) noexcept {
  const u64 ab = a < b ? a : b;
  const u64 cd = c < d ? c : d;
  return ab < cd ? ab : cd;
}

[[nodiscard]] constexpr ComputePlan
RejectCompute(const TilePhaseDescription &phase, const ComputeMap &map,
              const u64 bytes_per_tile, const u64 staging_bytes,
              const char *const reason) noexcept {
  return ComputePlan{
      .phase_id = phase.phase_id,
      .tile_count = phase.tile_count,
      .op_hash_hi = map.op_hash_hi,
      .op_hash_lo = map.op_hash_lo,
      .api = map.api,
      .scalar = map.scalar,
      .domain = map.domain,
      .fixed_format = map.fixed_format,
      .input_buffer_count = map.input_buffer_count,
      .output_buffer_count = map.output_buffer_count,
      .input_bytes_per_tile = map.input_bytes_per_tile,
      .output_bytes_per_tile = map.output_bytes_per_tile,
      .param_bytes = map.param_bytes,
      .metadata_bytes_per_tile = map.metadata_bytes_per_tile,
      .bytes_per_tile = bytes_per_tile,
      .staging_bytes = staging_bytes,
      .reason = reason,
  };
}

[[nodiscard]] constexpr ComputeDispatchPlan
RejectComputeDispatch(const u64 bytes_per_tile, const u64 staging_bytes,
                      const char *const reason) noexcept {
  return ComputeDispatchPlan{
      .bytes_per_tile = bytes_per_tile,
      .staging_bytes = staging_bytes,
      .reason = reason,
  };
}

} // namespace compute_detail

[[nodiscard]] constexpr bool ComputeScalarBytes(const ComputeScalar scalar,
                                                u64 &bytes) noexcept {
  switch (scalar) {
  case ComputeScalar::Lane32:
    bytes = 4u;
    return true;
  case ComputeScalar::Lane64:
    bytes = 8u;
    return true;
  }
  bytes = 0u;
  return false;
}

[[nodiscard]] constexpr bool
ComputeOutputBytesValid(const ComputeScalar scalar,
                        const u64 output_bytes) noexcept {
  u64 scalar_bytes = 0u;
  return ComputeScalarBytes(scalar, scalar_bytes) &&
         (output_bytes == scalar_bytes ||
          (scalar == ComputeScalar::Lane32 && output_bytes == sizeof(u64)) ||
          (scalar == ComputeScalar::Lane64 && output_bytes == sizeof(u32)));
}

[[nodiscard]] constexpr bool
ComputePlanScalarValid(const ComputePlan &plan) noexcept {
  u64 scalar_bytes = 0u;
  if (!ComputeScalarBytes(plan.scalar, scalar_bytes) ||
      plan.output_buffer_count == 0u || plan.param_bytes % scalar_bytes != 0u ||
      !checked::mul(plan.input_buffer_count, sizeof(u32)) ||
      !checked::mul(plan.input_buffer_count, scalar_bytes) ||
      !checked::mul(plan.output_buffer_count, scalar_bytes)) {
    return false;
  }
  const u64 minimum_input_bytes = plan.input_buffer_count * sizeof(u32);
  const u64 maximum_input_bytes = plan.input_buffer_count * scalar_bytes;
  const u64 full_output_bytes = plan.output_buffer_count * scalar_bytes;
  const bool output_ok =
      plan.output_bytes_per_tile == full_output_bytes ||
      (plan.output_buffer_count == 1u &&
       ComputeOutputBytesValid(plan.scalar, plan.output_bytes_per_tile));
  return output_ok && plan.input_bytes_per_tile >= minimum_input_bytes &&
         plan.input_bytes_per_tile <= maximum_input_bytes &&
         plan.input_bytes_per_tile % sizeof(u32) == 0u;
}

[[nodiscard]] constexpr bool
ComputePlanMatchesMap(const ComputePlan &plan, const ComputeMap &map) noexcept {
  return plan.op_hash_hi == map.op_hash_hi &&
         plan.op_hash_lo == map.op_hash_lo && plan.api == map.api &&
         plan.scalar == map.scalar && plan.domain == map.domain &&
         plan.fixed_format == map.fixed_format &&
         plan.input_buffer_count == map.input_buffer_count &&
         plan.output_buffer_count == map.output_buffer_count &&
         plan.input_bytes_per_tile == map.input_bytes_per_tile &&
         plan.output_bytes_per_tile == map.output_bytes_per_tile &&
         plan.param_bytes == map.param_bytes &&
         plan.metadata_bytes_per_tile == map.metadata_bytes_per_tile;
}

[[nodiscard]] constexpr bool
ComputePlanBytesValid(const ComputePlan &plan) noexcept {
  u64 bytes = plan.input_bytes_per_tile;
  if (!checked::add(bytes, plan.output_bytes_per_tile)) {
    return false;
  }
  bytes += plan.output_bytes_per_tile;
  if (!checked::add(bytes, plan.metadata_bytes_per_tile)) {
    return false;
  }
  bytes += plan.metadata_bytes_per_tile;
  if (bytes != plan.bytes_per_tile ||
      !checked::mul(plan.bytes_per_tile, plan.dispatch_window_tiles)) {
    return false;
  }
  const u64 tile_bytes = plan.bytes_per_tile * plan.dispatch_window_tiles;
  return checked::add(tile_bytes, plan.param_bytes) &&
         tile_bytes + plan.param_bytes == plan.staging_bytes;
}

[[nodiscard]] constexpr bool
ComputePlanShapeValid(const ComputePlan &plan) noexcept {
  return plan.ok && plan.fixed_authoritative &&
         (plan.op_hash_hi != 0u || plan.op_hash_lo != 0u) &&
         ComputeApiValid(plan.api) && ComputeScalarValid(plan.scalar) &&
         plan.tile_count != 0u && plan.dispatch_window_tiles != 0u &&
         plan.dispatch_window_tiles <= plan.tile_count &&
         plan.dispatch_count ==
             checked::ceil(plan.tile_count, plan.dispatch_window_tiles) &&
         plan.bytes_per_tile != 0u && plan.staging_bytes != 0u &&
         ComputePlanBytesValid(plan);
}

[[nodiscard]] constexpr ComputeDispatchPlan
PlanComputeDispatch(const TilePhaseDescription &phase, const ComputeMap &map,
                    const ComputeCaps &caps,
                    const ComputeLimit &limit) noexcept {
  const TilePhaseAdmissionResult phase_validation =
      ValidateTilePhaseDescription(phase);
  if (!phase_validation.ok) {
    return compute_detail::RejectComputeDispatch(0u, 0u,
                                                 "compute_phase_invalid");
  }

  if (!caps.ok) {
    return compute_detail::RejectComputeDispatch(0u, 0u,
                                                 "compute_caps_invalid");
  }
  if (map.op_hash_hi == 0u && map.op_hash_lo == 0u) {
    return compute_detail::RejectComputeDispatch(0u, 0u, "compute_op_invalid");
  }
  if (!ComputeScalarValid(map.scalar)) {
    return compute_detail::RejectComputeDispatch(0u, 0u,
                                                 "compute_scalar_unsupported");
  }
  if (!ComputeApiValid(map.api) || !ComputeApiValid(caps.api)) {
    return compute_detail::RejectComputeDispatch(0u, 0u,
                                                 "compute_api_unsupported");
  }
  if (map.api != caps.api) {
    return compute_detail::RejectComputeDispatch(0u, 0u,
                                                 "compute_backend_mismatch");
  }

  u64 bytes_per_tile = map.input_bytes_per_tile;
  if (!checked::add(bytes_per_tile, map.output_bytes_per_tile)) {
    return compute_detail::RejectComputeDispatch(bytes_per_tile, 0u,
                                                 "compute_workset_overflow");
  }
  bytes_per_tile += map.output_bytes_per_tile;
  if (!checked::add(bytes_per_tile, map.metadata_bytes_per_tile)) {
    return compute_detail::RejectComputeDispatch(bytes_per_tile, 0u,
                                                 "compute_workset_overflow");
  }
  bytes_per_tile += map.metadata_bytes_per_tile;

  if (bytes_per_tile == 0u) {
    return compute_detail::RejectComputeDispatch(0u, 0u,
                                                 "compute_workset_zero");
  }
  if (caps.max_window_tiles == 0u || limit.max_window_tiles == 0u) {
    return compute_detail::RejectComputeDispatch(bytes_per_tile, 0u,
                                                 "compute_window_zero");
  }

  if (!checked::add(bytes_per_tile, map.param_bytes)) {
    return compute_detail::RejectComputeDispatch(bytes_per_tile, 0u,
                                                 "compute_dispatch_overflow");
  }
  const u64 one_tile_staging = bytes_per_tile + map.param_bytes;
  if (caps.staging_bytes < one_tile_staging ||
      limit.staging_bytes < one_tile_staging) {
    return compute_detail::RejectComputeDispatch(
        bytes_per_tile, one_tile_staging, "compute_staging_insufficient");
  }

  const u64 staging_limit = caps.staging_bytes < limit.staging_bytes
                                ? caps.staging_bytes
                                : limit.staging_bytes;
  const u64 staging_tiles = (staging_limit - map.param_bytes) / bytes_per_tile;
  const u64 dispatch_window_tiles =
      compute_detail::Min4(caps.max_window_tiles, limit.max_window_tiles,
                           phase.tile_count, staging_tiles);
  if (dispatch_window_tiles == 0u) {
    return compute_detail::RejectComputeDispatch(
        bytes_per_tile, one_tile_staging, "compute_staging_insufficient");
  }

  if (!checked::mul(bytes_per_tile, dispatch_window_tiles)) {
    return compute_detail::RejectComputeDispatch(bytes_per_tile, 0u,
                                                 "compute_dispatch_overflow");
  }
  const u64 window_tile_bytes = bytes_per_tile * dispatch_window_tiles;
  if (!checked::add(window_tile_bytes, map.param_bytes)) {
    return compute_detail::RejectComputeDispatch(
        bytes_per_tile, window_tile_bytes, "compute_dispatch_overflow");
  }
  const u64 dispatch_staging = window_tile_bytes + map.param_bytes;
  if (caps.staging_bytes < dispatch_staging ||
      limit.staging_bytes < dispatch_staging) {
    return compute_detail::RejectComputeDispatch(
        bytes_per_tile, dispatch_staging, "compute_staging_insufficient");
  }

  const u64 dispatch_count =
      checked::ceil(phase.tile_count, dispatch_window_tiles);
  return ComputeDispatchPlan{
      .bytes_per_tile = bytes_per_tile,
      .staging_bytes = dispatch_staging,
      .dispatch_window_tiles = dispatch_window_tiles,
      .dispatch_count = dispatch_count,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr ComputePlan
PlanCompute(const TilePhaseDescription &phase, const ComputeMap &map,
            const ComputeCaps &caps, const ComputeLimit &limit) noexcept {
  const ComputeDispatchPlan dispatch =
      PlanComputeDispatch(phase, map, caps, limit);
  if (!dispatch.ok) {
    return compute_detail::RejectCompute(phase, map, dispatch.bytes_per_tile,
                                         dispatch.staging_bytes,
                                         dispatch.reason);
  }
  return ComputePlan{
      .phase_id = phase.phase_id,
      .tile_count = phase.tile_count,
      .op_hash_hi = map.op_hash_hi,
      .op_hash_lo = map.op_hash_lo,
      .api = map.api,
      .scalar = map.scalar,
      .domain = map.domain,
      .fixed_format = map.fixed_format,
      .input_buffer_count = map.input_buffer_count,
      .output_buffer_count = map.output_buffer_count,
      .input_bytes_per_tile = map.input_bytes_per_tile,
      .output_bytes_per_tile = map.output_bytes_per_tile,
      .param_bytes = map.param_bytes,
      .metadata_bytes_per_tile = map.metadata_bytes_per_tile,
      .bytes_per_tile = dispatch.bytes_per_tile,
      .staging_bytes = dispatch.staging_bytes,
      .dispatch_window_tiles = dispatch.dispatch_window_tiles,
      .dispatch_count = dispatch.dispatch_count,
      .fixed_authoritative = true,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace rund::kernel
