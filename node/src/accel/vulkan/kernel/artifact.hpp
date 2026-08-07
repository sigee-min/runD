#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>

#include <cstdint>
#include <string>
#include <utility>

namespace rund::node::accel::detail {

// Collective primitives are backend-authored programs, not canonical Map
// artifacts. Their complete cache identity is nevertheless the same tuple:
// a non-default ArtifactKey plus the exact full source retained by the cache.
[[nodiscard]] inline rund::kernel::LoweringArtifact
MakeVulkanBackendArtifact(const rund::kernel::ComputePlan &plan,
                          std::string source,
                          const std::uint64_t source_upper_bytes) noexcept {
  const rund::kernel::ArtifactKey key{
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = plan.scalar,
      .domain = plan.domain,
      .variant = rund::kernel::LoweringArtifactVariant::Canonical,
      .fixed_format = plan.fixed_format,
      .op_hash_hi = plan.op_hash_hi,
      .op_hash_lo = plan.op_hash_lo,
      .canonical_ir_hash_hi = plan.op_hash_hi,
      .canonical_ir_hash_lo = plan.op_hash_lo,
  };
  const bool valid = plan.ok && plan.api == rund::kernel::ComputeApi::Vulkan &&
                     !source.empty() && source_upper_bytes >= source.size();
  return rund::kernel::LoweringArtifact{
      .key = key,
      .kind = rund::kernel::LoweringArtifactKind::VulkanSource,
      .source_text = std::move(source),
      .source_text_upper_bytes = source_upper_bytes,
      .ok = valid,
      .reason = valid ? "ok" : "compute_artifact_mismatch",
  };
}

} // namespace rund::node::accel::detail
