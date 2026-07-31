#pragma once

#include "binding.hpp"
#include "param.hpp"
#include "read.hpp"
#include "output.hpp"

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] const char *
ValidateBindings(const rund::kernel::ComputeIR &ir, const ParsedIR &parsed,
                 const BindingPlan &plan,
                 const rund::kernel::BindingSet &bindings) {
  if (!bindings.ok) {
    return bindings.reason;
  }
  if (bindings.sequence_tiles != nullptr ||
      bindings.sequence_tile_count != 0u) {
    return "cpu_simd_sequence_unsupported";
  }
  if (bindings.tile_count == 0u) {
    return "cpu_simd_tile_count_invalid";
  }
  if (bindings.scalar != ir.scalar) {
    return "cpu_simd_scalar_mismatch";
  }
  if (bindings.domain != ir.domain) {
    return "cpu_simd_domain_mismatch";
  }
  if (bindings.op_hash_hi != ir.op_hash_hi ||
      bindings.op_hash_lo != ir.op_hash_lo) {
    return "cpu_simd_binding_artifact_mismatch";
  }

  const u64 scalar_bytes = ScalarBytes(ir.scalar);
  if (scalar_bytes == 0u) {
    return "cpu_simd_scalar_unsupported";
  }
  if (const char *const reason = ValidateParamBytes(parsed, plan, bindings);
      reason != nullptr) {
    return reason;
  }
  if (const char *const reason =
          ValidateReadBuffers(parsed, plan, bindings, scalar_bytes);
      reason != nullptr) {
    return reason;
  }
  return ValidateOutput(plan, bindings, scalar_bytes);
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
