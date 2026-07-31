#pragma once

#include <accel/runtime.hpp>

#include <accel/run/policy/choose.hpp>

#include "plan.hpp"

namespace node_accel_contract::policy_case {

[[nodiscard]] inline bool PlanRejected(const rund::kernel::ComputePlan &plan,
                                       const rund::RuntimeStats &evidence) {
  const rund::node::accel::RunChoice choice =
      rund::node::accel::ChooseRun(plan, evidence, {});
  return !choice.use_accel && ReasonIs(choice.reason, "accel_run_plan_invalid");
}

[[nodiscard]] inline bool
MalformedPlansReject(const rund::kernel::ComputePlan &valid,
                     const rund::RuntimeStats &evidence) {
  rund::kernel::ComputePlan invalid = valid;
  invalid.ok = false;
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.dispatch_count = 0u;
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.api = static_cast<rund::kernel::ComputeApi>(0u);
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.scalar = static_cast<rund::kernel::ComputeScalar>(0u);
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.op_hash_hi = 0u;
  invalid.op_hash_lo = 0u;
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.dispatch_window_tiles = valid.tile_count + 1u;
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.dispatch_count = valid.dispatch_count + 1u;
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.bytes_per_tile = valid.bytes_per_tile + 1u;
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.staging_bytes = valid.staging_bytes + 1u;
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }

  invalid = valid;
  invalid.input_bytes_per_tile = kMaxU64;
  invalid.output_bytes_per_tile = 1u;
  invalid.metadata_bytes_per_tile = 0u;
  invalid.bytes_per_tile = 1u;
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.input_bytes_per_tile = (kMaxU64 / 2u) + 1u;
  invalid.output_bytes_per_tile = 0u;
  invalid.metadata_bytes_per_tile = 0u;
  invalid.bytes_per_tile = invalid.input_bytes_per_tile;
  invalid.dispatch_window_tiles = 2u;
  invalid.dispatch_count = 4u;
  invalid.staging_bytes = 1u;
  if (!PlanRejected(invalid, evidence)) {
    return false;
  }
  invalid = valid;
  invalid.input_bytes_per_tile = kMaxU64;
  invalid.output_bytes_per_tile = 0u;
  invalid.metadata_bytes_per_tile = 0u;
  invalid.bytes_per_tile = kMaxU64;
  invalid.dispatch_window_tiles = 1u;
  invalid.dispatch_count = 8u;
  invalid.param_bytes = 1u;
  invalid.staging_bytes = kMaxU64;
  return PlanRejected(invalid, evidence);
}

} // namespace node_accel_contract::policy_case
