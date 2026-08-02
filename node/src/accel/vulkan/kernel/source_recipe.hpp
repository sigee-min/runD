#pragma once

#include "../../kernel/backend/source_recipe.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace rund::node::accel::detail {

// Small string-like adapter for migrating legacy append builders without
// keeping a second count formula. It records the first sink failure and makes
// decimal spelling consume the common charconv recipe rather than allocating
// a temporary std::string during planning.
template <typename Sink> class VulkanSourceTextSink final {
public:
  explicit VulkanSourceTextSink(Sink &sink) noexcept : sink_{sink} {}

  VulkanSourceTextSink &operator+=(const std::string_view fragment) noexcept(
      noexcept(sink_.append(fragment))) {
    if (ok_) {
      ok_ = sink_.append(fragment);
    }
    return *this;
  }

  void append(const std::string_view fragment) noexcept(
      noexcept(sink_.append(fragment))) {
    *this += fragment;
  }

  void decimal(const std::uint64_t value) noexcept(
      noexcept(sink_.append(std::string_view{}))) {
    if (ok_) {
      ok_ = backend_source_recipe::append_decimal(sink_, value);
    }
  }

  [[nodiscard]] bool ok() const noexcept { return ok_; }

private:
  Sink &sink_;
  bool ok_{true};
};

// Collective primitives are backend-authored programs, not canonical Map
// artifacts. Their complete cache identity is nevertheless the same tuple:
// a non-default ArtifactKey plus the exact full source retained by the cache.
[[nodiscard]] inline rund::kernel::ArtifactKey
VulkanBackendArtifactKey(const rund::kernel::ComputePlan &plan) noexcept {
  return rund::kernel::ArtifactKey{
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
}

[[nodiscard]] inline rund::kernel::LoweringArtifact
VulkanBackendArtifact(const rund::kernel::ComputePlan &plan, std::string source,
                      const std::uint64_t source_upper_bytes) noexcept {
  const bool valid = plan.ok && plan.api == rund::kernel::ComputeApi::Vulkan &&
                     !source.empty() && source_upper_bytes >= source.size();
  return rund::kernel::LoweringArtifact{
      .key = VulkanBackendArtifactKey(plan),
      .kind = rund::kernel::LoweringArtifactKind::VulkanSource,
      .source_text = std::move(source),
      .source_text_upper_bytes = source_upper_bytes,
      .ok = valid,
      .reason = valid ? "ok" : "compute_artifact_mismatch",
  };
}

} // namespace rund::node::accel::detail
