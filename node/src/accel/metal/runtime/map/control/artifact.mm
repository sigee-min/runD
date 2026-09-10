#include "../control.hpp"
#include "../source/upper.hpp"

#include "../../../../kernel/backend/exception.hpp"
#include "../../../../kernel/backend/source/storage.hpp"
#include "../../../../kernel/preparation.hpp"
#include "../../../pipeline/guard.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::kernel::LoweringArtifact
MetalControlledMapArtifact(rund::kernel::LoweringArtifact artifact,
                           const rund::kernel::ComputePlan &plan) {
  std::uint64_t source_upper = 0u;
  if (!MetalControlledMapSourceUpperBytes(
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
    using namespace metal_controlled_map_source_detail;
    if (artifact.source_text.capacity() < source_upper &&
        !backend_source_recipe::reserve_string(artifact.source_text,
                                               source_upper)) {
      artifact.ok = false;
      artifact.reason = "compute_pipeline_capacity";
      return artifact;
    }
    std::uint64_t binding = 0u;
    constexpr std::size_t ControlledCapacity =
        ControlledPrefix.size() + 20u + ControlledSuffix.size();
    std::array<char, ControlledCapacity> controlled_storage{};
    backend_source_recipe::FixedBufferSink<ControlledCapacity> controlled_sink{
        controlled_storage};
    constexpr std::size_t FunctionCapacity = FunctionPrefix.size() + 33u;
    std::array<char, FunctionCapacity> function_storage{};
    backend_source_recipe::FixedBufferSink<FunctionCapacity> function_sink{
        function_storage};
    if (!rund::kernel::checked::add(plan.input_buffer_count,
                                    plan.output_buffer_count, binding) ||
        !rund::kernel::checked::add(binding, 1u, binding) ||
        !controlled_sink.append(ControlledPrefix) ||
        !backend_source_recipe::append_decimal(controlled_sink, binding) ||
        !controlled_sink.append(ControlledSuffix) ||
        !function_sink.append(FunctionPrefix) ||
        !backend_source_recipe::append_hex64_digits(function_sink,
                                                    artifact.key.op_hash_hi) ||
        !function_sink.append("_") ||
        !backend_source_recipe::append_hex64_digits(function_sink,
                                                    artifact.key.op_hash_lo)) {
      artifact.ok = false;
      artifact.reason = "compute_pipeline_capacity";
      return artifact;
    }
    const std::string_view controlled = controlled_sink.text();
    const std::string_view function = function_sink.text();
    const std::size_t at = artifact.source_text.find(Entry);
    const std::size_t function_at = artifact.source_text.find(function);
    const std::size_t variant_at = artifact.source_text.find(CanonicalVariant);
    if (at == std::string::npos ||
        artifact.source_text.find(Entry, at + Entry.size()) !=
            std::string::npos ||
        function_at == std::string::npos ||
        artifact.source_text.find(function, function_at + function.size()) !=
            std::string::npos ||
        variant_at == std::string::npos ||
        artifact.source_text.find(CanonicalVariant,
                                  variant_at + CanonicalVariant.size()) !=
            std::string::npos) {
      artifact.ok = false;
      artifact.reason = "compute_artifact_mismatch";
      return artifact;
    }
    const std::size_t frozen_capacity = artifact.source_text.capacity();
    artifact.source_text.replace(at, Entry.size(), controlled.data(),
                                 controlled.size());
    artifact.source_text.insert(function_at + function.size(),
                                FunctionSuffix.data(), FunctionSuffix.size());
    artifact.source_text.replace(variant_at, CanonicalVariant.size(),
                                 ControlledVariant.data(),
                                 ControlledVariant.size());
    if (artifact.source_text.size() > source_upper ||
        artifact.source_text.capacity() != frozen_capacity) {
      artifact.ok = false;
      artifact.reason = "compute_pipeline_capacity";
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

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
MetalMapCheckHash(const MetalMapTemplateResources &prepared,
                  const rund::kernel::BindingSet &bindings) noexcept {
  std::uint64_t hi = prepared.plan.op_hash_hi ^ 0x6d61702e63686563ull;
  std::uint64_t lo = prepared.plan.op_hash_lo ^ 0x6b2e696e64657800ull;
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  for (const MetalMapCheck check : prepared.checks) {
    const auto *const ref = bindings.resident_inputs.ref(check.binding);
    mix(hi, check.binding);
    mix(hi, check.limit);
    mix(lo, ref == nullptr ? 0u : ref->offset_bytes);
    mix(lo, ref == nullptr ? 0u : ref->stride_bytes);
  }
  return {hi, lo};
}

namespace {

struct MetalMapCheckSourceRecipe final {
  const MetalMapTemplateResources &prepared;
  const rund::kernel::BindingSet &bindings;
  std::uint64_t hash_hi{};
  std::uint64_t hash_lo{};

  template <typename Sink>
  [[nodiscard]] bool operator()(Sink &sink) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
    using namespace metal_map_source_detail;
    if (!sink.append(CheckPrefix) || !sink.append(CheckNamePrefix) ||
        !backend_source_recipe::append_hex64_digits(sink, hash_hi) ||
        !sink.append("_") ||
        !backend_source_recipe::append_hex64_digits(sink, hash_lo) ||
        !sink.append(CheckArguments)) {
      return false;
    }
    for (std::size_t index = 0u; index < prepared.checks.size(); ++index) {
      if (!sink.append(CheckBindingPrefix) ||
          !backend_source_recipe::append_decimal(sink, index) ||
          !sink.append(CheckBindingMiddle) ||
          !backend_source_recipe::append_decimal(sink, index + 2u) ||
          !sink.append(CheckBindingSuffix)) {
        return false;
      }
    }
    const std::uint64_t config_binding = prepared.checks.size() + 2u;
    const std::uint64_t status_binding = config_binding + 1u;
    if (!sink.append(CheckConfigPrefix) ||
        !backend_source_recipe::append_decimal(sink, config_binding) ||
        !sink.append(CheckConfigMiddle) ||
        !backend_source_recipe::append_decimal(sink, status_binding) ||
        !sink.append(CheckBody)) {
      return false;
    }
    for (std::size_t index = 0u; index < prepared.checks.size(); ++index) {
      const MetalMapCheck check = prepared.checks[index];
      const auto *const ref = bindings.resident_inputs.ref(check.binding);
      if (ref == nullptr || !sink.append(CheckLinePrefix) ||
          !backend_source_recipe::append_decimal(sink, index) ||
          !sink.append(CheckLineStride) ||
          !backend_source_recipe::append_decimal(sink, ref->stride_bytes) ||
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

rund::kernel::LoweringArtifact
MetalMapCheckArtifact(const MetalMapTemplateResources &prepared,
                      const rund::kernel::BindingSet &bindings) {
  const auto [hi, lo] = MetalMapCheckHash(prepared, bindings);
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Metal;
  artifact.key.scalar = prepared.plan.scalar;
  artifact.key.domain = prepared.plan.domain;
  artifact.key.fixed_format = prepared.plan.fixed_format;
  artifact.key.op_hash_hi = hi;
  artifact.key.op_hash_lo = lo;
  artifact.key.canonical_ir_hash_hi = hi;
  artifact.key.canonical_ir_hash_lo = lo;
  artifact.kind = rund::kernel::LoweringArtifactKind::MetalSource;
  if (prepared.checks.empty() ||
      prepared.checks.size() > rund::kernel::kMaxComputeBindingCount) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  std::uint64_t stride_digits = 0u;
  std::uint64_t limit_digits = 0u;
  for (const MetalMapCheck check : prepared.checks) {
    const auto *const ref = bindings.resident_inputs.ref(check.binding);
    if (ref == nullptr ||
        !rund::kernel::checked::add(stride_digits,
                                    DecimalDigitCount(ref->stride_bytes),
                                    stride_digits) ||
        !rund::kernel::checked::add(
            limit_digits, DecimalDigitCount(check.limit), limit_digits)) {
      artifact.reason = "compute_pipeline_capacity";
      return artifact;
    }
  }
  std::uint64_t source_upper = 0u;
  std::uint64_t final_upper = 0u;
  if (!MetalMapCheckSourceUpperBytes(prepared.checks.size(), stride_digits,
                                     limit_digits, source_upper) ||
      !PipelinePrivateMetalSourceUpperBytes(
          source_upper, 1u,
          IsPipelinePrivatePreparation(CurrentKernelPreparationMode()),
          final_upper) ||
      final_upper > std::numeric_limits<std::size_t>::max()) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  const MetalMapCheckSourceRecipe recipe{prepared, bindings, hi, lo};
  std::uint64_t exact_source_bytes = 0u;
  if (!backend_source_recipe::bytes(recipe, exact_source_bytes) ||
      exact_source_bytes != source_upper) {
    artifact.reason = "compute_artifact_mismatch";
    return artifact;
  }
  artifact.source_text = backend_source_recipe::materialize(
      recipe, exact_source_bytes, final_upper);
  if (artifact.source_text.empty()) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  // This artifact still owns the raw check source. MetalPipelineForArtifact
  // is the sole Pipeline-private guard materializer and adds that ABI upper
  // exactly once; retaining final_upper here double-reserved every check
  // library on the cold path.
  artifact.source_text_upper_bytes = source_upper;
  artifact.ok = true;
  artifact.reason = "ok";
  return artifact;
}

#endif

} // namespace rund::node::accel::detail
