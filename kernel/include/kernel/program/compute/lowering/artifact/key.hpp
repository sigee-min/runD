#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/model.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] inline ArtifactKey
MakeKey(const ComputeIR &ir, const ComputeApi api,
        const std::vector<u8> &canonical_bytes) {
  const compute_ir_detail::ComputeIrHash canonical_hash =
      compute_ir_detail::HashComputeIrCanonicalBytes(
          canonical_bytes.empty() ? nullptr : canonical_bytes.data(),
          static_cast<u64>(canonical_bytes.size()));
  return ArtifactKey{
      .api = api,
      .scalar = ir.scalar,
      .domain = ir.domain,
      .fixed_format = ir.fixed_format,
      .op_hash_hi = ir.op_hash_hi,
      .op_hash_lo = ir.op_hash_lo,
      .canonical_ir_hash_hi = canonical_hash.hi,
      .canonical_ir_hash_lo = canonical_hash.lo,
  };
}

[[nodiscard]] inline ArtifactKey MakeKey(const ComputeIR &ir,
                                         const ComputeApi api) {
  return MakeKey(ir, api, ir.canonical_bytes);
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
