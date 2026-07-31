#include "plan/local.hpp"

namespace program_compute_contract {

int RunComputePlanContract() {
  if (test_compute_plan_rejects_invalid_phase() != 0) {
    return 1;
  }
  if (test_compute_plan_rejects_zero_workset() != 0) {
    return 1;
  }
  if (test_compute_plan_rejects_missing_op_hash() != 0) {
    return 1;
  }
  if (test_compute_plan_rejects_overflow_workset() != 0) {
    return 1;
  }
  if (test_compute_plan_rejects_staging_insufficient() != 0) {
    return 1;
  }
  if (test_compute_plan_chunks_to_staging_capacity() != 0) {
    return 1;
  }
  if (test_compute_plan_rejects_zero_dispatch_limit() != 0) {
    return 1;
  }
  if (test_compute_plan_rejects_missing_caps() != 0) {
    return 1;
  }
  if (test_compute_plan_rejects_nonfixed_scalar() != 0) {
    return 1;
  }
  if (test_compute_plan_rejects_backend_mismatch() != 0) {
    return 1;
  }
  if (test_compute_plan_rejects_unknown_api_even_when_caps_match() != 0) {
    return 1;
  }
  if (test_compute_plan_is_deterministic_for_same_inputs() != 0) {
    return 1;
  }
  if (test_compute_plan_computes_dispatch_windows() != 0) {
    return 1;
  }
  if (test_compute_plan_shape_guard_rejects_forgery() != 0) {
    return 1;
  }
  if (test_compute_plan_scalar_guard_is_separate() != 0) {
    return 1;
  }
  if (test_compute_dispatch_plan_matches_full_plan() != 0) {
    return 1;
  }
  if (test_compute_dispatch_plan_rejects_like_full_plan() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
