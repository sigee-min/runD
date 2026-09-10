#include "control.hpp"

#include "../../../kernel/backend/exception.hpp"
#include "../../../kernel/backend/source/storage.hpp"
#include "../local.hpp"
#include "upper.hpp"
#include "upper/check_fragments.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

struct VulkanMapCheckSourceRecipe final {
  const VulkanMapTemplateResources &prepared;

  template <typename Sink>
  [[nodiscard]] bool operator()(Sink &sink) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
    using namespace vulkan_map_source_detail;
    if (!sink.append(CheckPrefix)) {
      return false;
    }
    for (std::size_t index = 0u; index < prepared.checks.size(); ++index) {
      if (!sink.append(BindingPrefix) ||
          !backend_source_recipe::append_decimal(sink, index + 2u) ||
          !sink.append(BindingIndex) ||
          !backend_source_recipe::append_decimal(sink, index) ||
          !sink.append(BindingWords) ||
          !backend_source_recipe::append_decimal(sink, index) ||
          !sink.append(BindingSuffix)) {
        return false;
      }
    }
    if (!sink.append(BindingPrefix) ||
        !backend_source_recipe::append_decimal(sink,
                                               prepared.checks.size() + 2u) ||
        !sink.append(StatusMiddle) || !sink.append(CheckBody)) {
      return false;
    }
    for (std::size_t index = 0u; index < prepared.checks.size(); ++index) {
      const VulkanMapCheck check = prepared.checks[index];
      if (!sink.append(CheckLinePrefix) ||
          !backend_source_recipe::append_decimal(sink, index) ||
          !sink.append(CheckLineOffset) ||
          !backend_source_recipe::append_decimal(sink, check.offset) ||
          !sink.append(CheckLineStride) ||
          !backend_source_recipe::append_decimal(sink, check.stride) ||
          !sink.append(CheckLineLimit) ||
          !backend_source_recipe::append_decimal(sink, check.limit) ||
          !sink.append(CheckLineSuffix)) {
        return false;
      }
    }
    return sink.append(CheckTail);
  }
};

} // namespace

std::pair<std::uint64_t, std::uint64_t>
VulkanMapCheckHash(const VulkanMapTemplateResources &prepared) noexcept {
  std::uint64_t hi = prepared.plan.op_hash_hi ^ 0x6d61702e63686563ull;
  std::uint64_t lo = prepared.plan.op_hash_lo ^ 0x6b2e696e64657800ull;
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  for (const VulkanMapCheck check : prepared.checks) {
    mix(hi, check.binding);
    mix(hi, check.limit);
    mix(lo, check.offset);
    mix(lo, check.stride);
  }
  return {hi, lo};
}

rund::kernel::LoweringArtifact
VulkanMapCheckArtifact(const VulkanMapTemplateResources &prepared) {
  const auto [hi, lo] = VulkanMapCheckHash(prepared);
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Vulkan;
  artifact.key.scalar = prepared.plan.scalar;
  artifact.key.domain = prepared.plan.domain;
  artifact.key.fixed_format = prepared.plan.fixed_format;
  artifact.key.op_hash_hi = hi;
  artifact.key.op_hash_lo = lo;
  artifact.key.canonical_ir_hash_hi = hi;
  artifact.key.canonical_ir_hash_lo = lo;
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  if (prepared.checks.empty() ||
      prepared.checks.size() > rund::kernel::kMaxComputeBindingCount) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  std::uint64_t offset_digits = 0u;
  std::uint64_t stride_digits = 0u;
  std::uint64_t limit_digits = 0u;
  for (const VulkanMapCheck check : prepared.checks) {
    if (!rund::kernel::checked::add(offset_digits,
                                    VulkanDecimalDigitCount(check.offset),
                                    offset_digits) ||
        !rund::kernel::checked::add(stride_digits,
                                    VulkanDecimalDigitCount(check.stride),
                                    stride_digits) ||
        !rund::kernel::checked::add(
            limit_digits, VulkanDecimalDigitCount(check.limit), limit_digits)) {
      artifact.reason = "compute_pipeline_capacity";
      return artifact;
    }
  }
  std::uint64_t source_upper = 0u;
  if (!VulkanMapCheckSourceUpperBytes(prepared.checks.size(), offset_digits,
                                      stride_digits, limit_digits,
                                      source_upper) ||
      source_upper > std::numeric_limits<std::size_t>::max()) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  const VulkanMapCheckSourceRecipe recipe{prepared};
  std::uint64_t exact_source_bytes = 0u;
  if (!backend_source_recipe::bytes(recipe, exact_source_bytes) ||
      exact_source_bytes != source_upper) {
    artifact.reason = "compute_artifact_mismatch";
    return artifact;
  }
  artifact.source_text =
      backend_source_recipe::materialize(recipe, exact_source_bytes);
  if (artifact.source_text.empty()) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  artifact.source_text_upper_bytes = source_upper;
  artifact.ok = true;
  artifact.reason = "ok";
  return artifact;
}

rund::kernel::LoweringArtifact
VulkanMapControlArtifact(const rund::kernel::ComputePlan &plan) {
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Vulkan;
  artifact.key.scalar = plan.scalar;
  artifact.key.domain = plan.domain;
  artifact.key.fixed_format = plan.fixed_format;
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  std::uint64_t source_bytes = 0u;
  if (!VulkanMapControlSourceBytes(source_bytes)) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  artifact.source_text = VulkanMapControlSource();
  artifact.source_text_upper_bytes = source_bytes;
  if (artifact.source_text.empty()) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  artifact.ok = artifact.source_text.size() == source_bytes;
  artifact.reason = artifact.ok ? "ok" : "compute_artifact_mismatch";
  return artifact;
}

rund::kernel::LoweringArtifact
VulkanGeneratedMapControlArtifact(const rund::kernel::ComputePlan &plan) {
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Vulkan;
  artifact.key.scalar = plan.scalar;
  artifact.key.domain = plan.domain;
  artifact.key.fixed_format = plan.fixed_format;
  artifact.key.variant = rund::kernel::LoweringArtifactVariant::Controlled;
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  std::uint64_t source_bytes = 0u;
  if (!VulkanGeneratedMapControlSourceBytes(source_bytes)) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  artifact.source_text = VulkanGeneratedMapControlSource();
  artifact.source_text_upper_bytes = source_bytes;
  if (artifact.source_text.empty()) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  artifact.ok = artifact.source_text.size() == source_bytes;
  artifact.reason = artifact.ok ? "ok" : "compute_artifact_mismatch";
  return artifact;
}

rund::kernel::LoweringArtifact
VulkanControlledMapArtifact(rund::kernel::LoweringArtifact artifact,
                            const rund::kernel::ComputePlan &plan) {
  std::uint64_t source_upper = 0u;
  if (!VulkanControlledMapSourceUpperBytes(
          plan,
          std::max<std::uint64_t>(artifact.source_text.size(),
                                  artifact.source_text_upper_bytes),
          source_upper) ||
      source_upper > std::numeric_limits<std::size_t>::max()) {
    artifact.ok = false;
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  try {
    using namespace vulkan_controlled_map_source_detail;
    if (!backend_source_recipe::reserve_string(artifact.source_text,
                                               source_upper)) {
      artifact.ok = false;
      artifact.reason = "compute_pipeline_capacity";
      return artifact;
    }
    std::uint64_t binding = 0u;
    constexpr std::size_t ControlledEntryCapacity =
        DeclarationPrefix.size() + 20u + DeclarationSuffix.size() +
        Entry.size();
    std::array<char, ControlledEntryCapacity> controlled_storage{};
    backend_source_recipe::FixedBufferSink<ControlledEntryCapacity>
        controlled_sink{controlled_storage};
    if (!rund::kernel::checked::add(plan.input_buffer_count,
                                    plan.output_buffer_count, binding) ||
        !rund::kernel::checked::add(binding, 1u, binding) ||
        !controlled_sink.append(DeclarationPrefix) ||
        !backend_source_recipe::append_decimal(controlled_sink, binding) ||
        !controlled_sink.append(DeclarationSuffix) ||
        !controlled_sink.append(Entry)) {
      artifact.ok = false;
      artifact.reason = "compute_pipeline_capacity";
      return artifact;
    }
    const std::size_t frozen_capacity = artifact.source_text.capacity();
    const auto replace_unique = [&](const std::string_view needle,
                                    const std::string_view replacement) {
      const std::size_t at = artifact.source_text.find(needle);
      if (needle.empty() || at == std::string::npos ||
          artifact.source_text.find(needle, at + needle.size()) !=
              std::string::npos) {
        return false;
      }
      artifact.source_text.replace(at, needle.size(), replacement.data(),
                                   replacement.size());
      return artifact.source_text.capacity() == frozen_capacity;
    };
    if (!replace_unique(Entry, controlled_sink.text()) ||
        !replace_unique(Guard, ControlledGuard) ||
        !replace_unique(CanonicalVariant, ControlledVariant) ||
        artifact.source_text.size() > source_upper ||
        artifact.source_text.capacity() != frozen_capacity) {
      artifact.ok = false;
      artifact.reason = "compute_artifact_mismatch";
      return artifact;
    }
    artifact.key.variant = rund::kernel::LoweringArtifactVariant::Controlled;
    artifact.source_text_upper_bytes = source_upper;
    return artifact;
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    artifact.ok = false;
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
}

#endif

} // namespace rund::node::accel::detail
