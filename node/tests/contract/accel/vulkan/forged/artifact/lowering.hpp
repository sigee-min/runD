#pragma once

#include <accel/device.hpp>

#include "op.hpp"

namespace node_accel_contract::vulkan {

struct ForgedArtifactLowering {
  rund::kernel::ComputeMap map{};
  rund::kernel::ComputePlan plan{};
  rund::kernel::LoweringArtifact artifact{};
  bool ok = false;
};

[[nodiscard]] inline bool
MutateToValidArtifactKey(rund::kernel::LoweringArtifact &forged,
                         const rund::kernel::LoweringArtifact &valid) {
  if (!ReplaceAll(forged.source_text,
                  KeyLine("op_hash_hi", forged.key.op_hash_hi),
                  KeyLine("op_hash_hi", valid.key.op_hash_hi))) {
    return false;
  }
  if (!ReplaceAll(forged.source_text,
                  KeyLine("op_hash_lo", forged.key.op_hash_lo),
                  KeyLine("op_hash_lo", valid.key.op_hash_lo))) {
    return false;
  }
  if (!ReplaceAll(
          forged.source_text,
          KeyLine("canonical_ir_hash_hi", forged.key.canonical_ir_hash_hi),
          KeyLine("canonical_ir_hash_hi", valid.key.canonical_ir_hash_hi))) {
    return false;
  }
  return ReplaceAll(
      forged.source_text,
      KeyLine("canonical_ir_hash_lo", forged.key.canonical_ir_hash_lo),
      KeyLine("canonical_ir_hash_lo", valid.key.canonical_ir_hash_lo));
}

[[nodiscard]] inline ForgedArtifactLowering
BuildForgedArtifactLowering(const rund::AccelDevice &pick,
                            const rund::compute_dsl::ComputeOp &valid_op,
                            const rund::compute_dsl::ComputeOp &forged_op) {
  ForgedArtifactLowering out{};
  out.map = valid_op.map();
  out.map.api = rund::kernel::ComputeApi::Vulkan;
  out.plan = rund::kernel::PlanCompute(
      Phase(), out.map, pick.caps,
      rund::kernel::ComputeLimit{
          .staging_bytes = pick.caps.staging_bytes,
          .max_window_tiles = pick.caps.max_window_tiles,
      });
  if (!out.plan.ok || !pick.backend) {
    return out;
  }

  const rund::kernel::LoweringArtifact valid = rund::kernel::LowerComputeIR(
      valid_op.ir(), out.plan.api);
  out.artifact = rund::kernel::LowerComputeIR(forged_op.ir(), out.plan.api);
  if (!valid.ok || !out.artifact.ok ||
      !MutateToValidArtifactKey(out.artifact, valid)) {
    return out;
  }
  out.artifact.key = valid.key;
  out.ok = true;
  return out;
}

} // namespace node_accel_contract::vulkan
