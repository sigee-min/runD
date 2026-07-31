#pragma once

#include <kernel/program/compute/metadata.hpp>
#include <kernel/program/compute/retention.hpp>

#include <string>
#include <vector>

namespace rund::kernel {

enum class LoweringArtifactKind : u8 {
  None = 0u,
  MetalSource = 1u,
  VulkanIdentity = 2u,
  VulkanSource = 3u,
  CpuPlan = 4u,
};

enum class LoweringArtifactVariant : u8 {
  Canonical = 0u,
  Controlled = 1u,
  Recurrence = 2u,
};

struct ArtifactKey {
  ComputeApi api = ComputeApi::Metal;
  ComputeScalar scalar = ComputeScalar::Lane32;
  ComputeDomain domain = ComputeDomain::Fixed;
  LoweringArtifactVariant variant = LoweringArtifactVariant::Canonical;
  ComputeFixedFormat fixed_format{};
  u64 op_hash_hi = 0u;
  u64 op_hash_lo = 0u;
  u64 canonical_ir_hash_hi = 0u;
  u64 canonical_ir_hash_lo = 0u;

  [[nodiscard]] friend constexpr bool
  operator==(const ArtifactKey &lhs, const ArtifactKey &rhs) noexcept {
    return lhs.api == rhs.api && lhs.scalar == rhs.scalar &&
           lhs.domain == rhs.domain && lhs.variant == rhs.variant &&
           lhs.fixed_format == rhs.fixed_format &&
           lhs.op_hash_hi == rhs.op_hash_hi &&
           lhs.op_hash_lo == rhs.op_hash_lo &&
           lhs.canonical_ir_hash_hi == rhs.canonical_ir_hash_hi &&
           lhs.canonical_ir_hash_lo == rhs.canonical_ir_hash_lo;
  }

  [[nodiscard]] friend constexpr bool
  operator!=(const ArtifactKey &lhs, const ArtifactKey &rhs) noexcept {
    return !(lhs == rhs);
  }
};

struct LoweringArtifact {
  ArtifactKey key{};
  LoweringArtifactKind kind = LoweringArtifactKind::None;
  ExecutionMetadata metadata{};
  std::string source_text;
  std::vector<u8> canonical_ir_bytes;
  bool ok = false;
  const char *reason = "compute_lowering_invalid";

  [[nodiscard]] explicit operator bool() const noexcept { return ok; }

  // Counts allocations owned below this object. The inline object itself is
  // counted by its enclosing owner.
  [[nodiscard]] u64 retained_dynamic_memory_bytes() const noexcept {
    using compute_retained_detail::Add;
    using compute_retained_detail::StringExternalStorageBytes;
    using compute_retained_detail::VectorCapacityBytes;
    u64 retained = metadata.retained_dynamic_memory_bytes();
    retained = Add(retained, StringExternalStorageBytes(source_text));
    return Add(retained, VectorCapacityBytes(canonical_ir_bytes));
  }
};

} // namespace rund::kernel
