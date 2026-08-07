#pragma once

#include "../../backend/exception.hpp"
#include "../../backend/source_recipe.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/limit.hpp>
#include <kernel/program/compute/lowering/text.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

namespace rund::node::accel::detail {

// The canonical Map emitter freezes its own exact-IR source upper. Binding
// specialization can replace one base and one stride decimal literal per
// binding. A U64 replacement can grow a one-digit canonical literal by at
// most 19 bytes; this allocation-free owner is shared by public planning and
// the runtime mutator below.
[[nodiscard]] inline bool MapSpecializedSourceUpperBytes(
    const std::uint64_t source_bytes, const std::uint64_t source_upper_bytes,
    const rund::kernel::ComputePlan &plan, std::uint64_t &upper) noexcept {
  constexpr std::uint64_t DecimalWidth =
      std::numeric_limits<std::uint64_t>::digits10 + 1u;
  constexpr std::uint64_t LiteralGrowth = DecimalWidth - 1u;
  std::uint64_t binding_count = 0u;
  std::uint64_t growth = 0u;
  upper = std::max(source_bytes, source_upper_bytes);
  return rund::kernel::checked::add(plan.input_buffer_count,
                                    plan.output_buffer_count, binding_count) &&
         rund::kernel::checked::mul(binding_count, 2u * LiteralGrowth,
                                    growth) &&
         rund::kernel::checked::add(upper, growth, upper);
}

[[nodiscard]] inline bool
MapSpecializedSourceUpperBytes(const rund::kernel::LoweringArtifact &source,
                               const rund::kernel::ComputePlan &plan,
                               std::uint64_t &upper) noexcept {
  return MapSpecializedSourceUpperBytes(
      source.source_text.size(), source.source_text_upper_bytes, plan, upper);
}

inline constexpr std::size_t MapSpecializationEditCapacity =
    2u * static_cast<std::size_t>(rund::kernel::kMaxComputeBindingCount);

struct MapSourceEdit final {
  std::size_t begin{};
  std::size_t end{};
  std::array<char, 20u> replacement{};
  std::uint8_t replacement_size{};

  [[nodiscard]] std::string_view text() const noexcept {
    return std::string_view{replacement.data(), replacement_size};
  }
};

[[nodiscard]] inline bool
ConsumeMapSourceFragment(const std::string_view source, std::size_t &cursor,
                         const std::string_view fragment) noexcept {
  if (cursor > source.size() || fragment.size() > source.size() - cursor ||
      source.compare(cursor, fragment.size(), fragment) != 0) {
    return false;
  }
  cursor += fragment.size();
  return true;
}

[[nodiscard]] inline bool
ConsumeMapSafeIdentifier(const std::string_view source, std::size_t &cursor,
                         const std::string_view name) noexcept {
  if (name.empty()) {
    return ConsumeMapSourceFragment(source, cursor, "empty");
  }
  for (const char value : name) {
    if (cursor > source.size() || source.size() - cursor < 2u) {
      return false;
    }
    const auto byte =
        static_cast<rund::kernel::u8>(static_cast<unsigned char>(value));
    if (source[cursor] !=
            rund::kernel::compute_lowering_detail::HexDigit(
                static_cast<rund::kernel::u8>((byte >> 4u) & 0x0fu)) ||
        source[cursor + 1u] !=
            rund::kernel::compute_lowering_detail::HexDigit(
                static_cast<rund::kernel::u8>(byte & 0x0fu))) {
      return false;
    }
    cursor += 2u;
  }
  return true;
}

[[nodiscard]] inline bool FindMapDeclarationValue(
    const std::string_view source, const std::string_view qualifier,
    const std::string_view symbol_prefix, const std::string_view access_prefix,
    const std::string_view binding_name, std::size_t &value_at) noexcept {
  value_at = std::string_view::npos;
  std::size_t search = 0u;
  while (search < source.size()) {
    const std::size_t at = source.find(qualifier, search);
    if (at == std::string_view::npos) {
      break;
    }
    std::size_t cursor = at;
    if (ConsumeMapSourceFragment(source, cursor, qualifier) &&
        ConsumeMapSourceFragment(source, cursor, symbol_prefix) &&
        ConsumeMapSourceFragment(source, cursor, access_prefix) &&
        ConsumeMapSafeIdentifier(source, cursor, binding_name) &&
        ConsumeMapSourceFragment(source, cursor, " = ")) {
      if (value_at != std::string_view::npos) {
        return false;
      }
      value_at = cursor;
    }
    search = at + 1u;
  }
  return value_at != std::string_view::npos;
}

[[nodiscard]] inline bool PlanMapDeclarationValue(
    const std::span<MapSourceEdit> edits, std::size_t &edit_count,
    const std::string_view source, const std::string_view qualifier,
    const std::string_view symbol_prefix, const std::string_view access_prefix,
    const std::string_view binding_name, const std::uint64_t expected,
    const std::uint64_t replacement) noexcept {
  if (edit_count == edits.size()) {
    return false;
  }
  std::array<char, 20u> expected_storage{};
  const std::string_view expected_text =
      backend_source_recipe::decimal_characters(expected, expected_storage);
  MapSourceEdit candidate{};
  const std::string_view replacement_text =
      backend_source_recipe::decimal_characters(replacement,
                                                candidate.replacement);
  std::size_t value_at = std::string_view::npos;
  if (expected_text.empty() || replacement_text.empty() ||
      !FindMapDeclarationValue(source, qualifier, symbol_prefix, access_prefix,
                               binding_name, value_at) ||
      value_at > source.size() ||
      expected_text.size() > source.size() - value_at ||
      source.compare(value_at, expected_text.size(), expected_text) != 0 ||
      source.size() - (value_at + expected_text.size()) < 2u ||
      source.compare(value_at + expected_text.size(), 2u, "u;") != 0) {
    return false;
  }
  candidate.begin = value_at;
  candidate.end = value_at + expected_text.size();
  candidate.replacement_size =
      static_cast<std::uint8_t>(replacement_text.size());
  edits[edit_count++] = candidate;
  return true;
}

struct MapSourceSpecialization final {
  std::array<MapSourceEdit, MapSpecializationEditCapacity> edits{};
  std::size_t edit_count{};
  std::uint64_t exact_source_bytes{};
  std::uint64_t source_upper_bytes{};
  std::uint64_t reserve_upper_bytes{};
  bool ok{};
  const char *reason{"compute_artifact_mismatch"};

  [[nodiscard]] std::span<const MapSourceEdit> active_edits() const noexcept {
    return {edits.data(), edit_count};
  }
};

[[nodiscard]] inline MapSourceSpecialization
PlanMapSourceSpecialization(const rund::kernel::LoweringArtifact &source,
                            const rund::kernel::ComputePlan &plan,
                            const rund::kernel::BindingSet &bindings,
                            const std::uint64_t alignment,
                            const std::uint64_t reserve_upper) noexcept {
  MapSourceSpecialization result{};
  const rund::kernel::ExecutionMetadata &metadata = source.metadata;
  const bool source_kind_matches =
      (source.key.api == rund::kernel::ComputeApi::Metal &&
       source.kind == rund::kernel::LoweringArtifactKind::MetalSource) ||
      (source.key.api == rund::kernel::ComputeApi::Vulkan &&
       source.kind == rund::kernel::LoweringArtifactKind::VulkanSource);
  if (!source.ok) {
    result.reason = source.reason;
    return result;
  }
  if (!source_kind_matches || source.key.api != plan.api ||
      source.source_text.empty() ||
      source.source_text_upper_bytes < source.source_text.size() ||
      !metadata.ok ||
      metadata.binding_names.size() != metadata.binding_accesses.size() ||
      alignment == 0u) {
    return result;
  }
  std::uint64_t binding_count = 0u;
  if (!MapSpecializedSourceUpperBytes(source, plan,
                                      result.source_upper_bytes) ||
      !rund::kernel::checked::add(plan.input_buffer_count,
                                  plan.output_buffer_count, binding_count) ||
      binding_count > rund::kernel::kMaxComputeBindingCount) {
    result.reason = "compute_pipeline_capacity";
    return result;
  }
  result.reserve_upper_bytes =
      std::max(result.source_upper_bytes, reserve_upper);
  if (result.reserve_upper_bytes > std::numeric_limits<std::size_t>::max()) {
    result.reason = "compute_pipeline_capacity";
    return result;
  }
  if (metadata.binding_accesses.size() !=
          static_cast<std::size_t>(binding_count) ||
      metadata.input_element_bytes.size() !=
          static_cast<std::size_t>(plan.input_buffer_count) ||
      metadata.output_element_bytes.size() !=
          static_cast<std::size_t>(plan.output_buffer_count)) {
    return result;
  }

  std::size_t read = 0u;
  std::size_t write = 0u;
  for (std::size_t index = 0u; index < metadata.binding_accesses.size();
       ++index) {
    const auto access = metadata.binding_accesses[index];
    const bool is_read = access == rund::kernel::ComputeBindingAccess::Read;
    const bool is_write = access == rund::kernel::ComputeBindingAccess::Write;
    if (!is_read && !is_write) {
      return result;
    }
    if ((is_read && read >= metadata.input_element_bytes.size()) ||
        (is_write && write >= metadata.output_element_bytes.size())) {
      return result;
    }
    const rund::kernel::ResidentBufferRef *const ref =
        is_read ? bindings.resident_inputs.ref(read)
                : bindings.resident_outputs.ref(write);
    const std::uint64_t element_bytes =
        is_read ? metadata.input_element_bytes[read]
                : metadata.output_element_bytes[write];
    read += is_read ? 1u : 0u;
    write += is_write ? 1u : 0u;
    if (ref == nullptr || ref->element_bytes != element_bytes ||
        ref->stride_bytes < element_bytes) {
      result.reason = "compute_resident_stride_invalid";
      return result;
    }
    const std::string_view qualifier =
        source.key.api == rund::kernel::ComputeApi::Metal
            ? std::string_view{"constant uint "}
            : std::string_view{"const uint "};
    const std::string_view access_prefix =
        is_read ? std::string_view{"read_"} : std::string_view{"write_"};
    if (ref->stride_bytes != element_bytes &&
        !PlanMapDeclarationValue(
            std::span<MapSourceEdit>{result.edits}, result.edit_count,
            source.source_text, qualifier, "RundStride_", access_prefix,
            metadata.binding_names[index], element_bytes, ref->stride_bytes)) {
      return result;
    }
    const std::uint64_t base = ref->offset_bytes % alignment;
    if (base != 0u &&
        !PlanMapDeclarationValue(std::span<MapSourceEdit>{result.edits},
                                 result.edit_count, source.source_text,
                                 qualifier, "RundBase_", access_prefix,
                                 metadata.binding_names[index], 0u, base)) {
      return result;
    }
  }
  if (read != plan.input_buffer_count || write != plan.output_buffer_count) {
    return result;
  }
  const std::span<MapSourceEdit> active_edits{result.edits.data(),
                                              result.edit_count};
  if (!active_edits.empty() && !backend_source_recipe::canonicalize_edits(
                                   active_edits, source.source_text.size())) {
    return result;
  }
  using Recipe = backend_source_recipe::BasicSourceEditRecipe<MapSourceEdit>;
  const Recipe recipe{source.source_text, result.active_edits()};
  if (!backend_source_recipe::bytes(recipe, result.exact_source_bytes) ||
      result.exact_source_bytes > result.source_upper_bytes) {
    return result;
  }
  result.ok = true;
  result.reason = "ok";
  return result;
}

[[nodiscard]] inline rund::kernel::LoweringArtifact
SpecializeMap(const rund::kernel::LoweringArtifact &source,
              const rund::kernel::ComputePlan &plan,
              const rund::kernel::BindingSet &bindings,
              const std::uint64_t alignment = 1u,
              const std::uint64_t reserve_upper = 0u) {
  // Specialization only mutates source text. Metadata and canonical IR remain
  // owned by the admitted Program; copying those vectors here would recreate
  // the cold intermediate layer removed by this boundary.
  const MapSourceSpecialization specialization = PlanMapSourceSpecialization(
      source, plan, bindings, alignment, reserve_upper);
  rund::kernel::LoweringArtifact artifact{
      .key = source.key,
      .kind = source.kind,
      .source_text_upper_bytes = specialization.source_upper_bytes,
      .ok = specialization.ok,
      .reason = specialization.reason,
  };
  if (!specialization.ok) {
    return artifact;
  }
  using Recipe = backend_source_recipe::BasicSourceEditRecipe<MapSourceEdit>;
  const Recipe recipe{source.source_text, specialization.active_edits()};
  artifact.source_text = backend_source_recipe::materialize(
      recipe, specialization.exact_source_bytes,
      specialization.reserve_upper_bytes);
  if (artifact.source_text.empty() ||
      artifact.source_text.size() != specialization.exact_source_bytes) {
    artifact.ok = false;
    artifact.reason = "compute_pipeline_capacity";
  }
  return artifact;
}

// Recurrence materialization owns its artifact outright and reserves the
// final backend source envelope before this call. Apply binding edits from
// right to left inside that sole allocation, then discard semantic metadata
// before compilation. Insufficient frozen capacity fails closed instead of
// allocating a second transformed source owner.
[[nodiscard]] inline rund::kernel::LoweringArtifact
SpecializeMapInPlace(rund::kernel::LoweringArtifact &&source,
                     const rund::kernel::ComputePlan &plan,
                     const rund::kernel::BindingSet &bindings,
                     const std::uint64_t alignment = 1u,
                     const std::uint64_t reserve_upper = 0u) {
  const MapSourceSpecialization specialization = PlanMapSourceSpecialization(
      source, plan, bindings, alignment, reserve_upper);
  if (!specialization.ok) {
    source.ok = false;
    source.reason = specialization.reason;
    return std::move(source);
  }
  std::uint64_t storage_upper = 0u;
  if (source.source_text.capacity() < specialization.reserve_upper_bytes ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          specialization.reserve_upper_bytes, storage_upper) ||
      !backend_source_recipe::string_external_storage_within(source.source_text,
                                                             storage_upper)) {
    source.ok = false;
    source.reason = "compute_pipeline_capacity";
    return std::move(source);
  }
  const std::size_t frozen_capacity = source.source_text.capacity();
  const char *const frozen_storage = source.source_text.data();
  try {
    const std::span<const MapSourceEdit> edits = specialization.active_edits();
    for (std::size_t reverse = edits.size(); reverse != 0u; --reverse) {
      const MapSourceEdit &edit = edits[reverse - 1u];
      source.source_text.replace(edit.begin, edit.end - edit.begin,
                                 edit.replacement.data(),
                                 edit.replacement_size);
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    source.ok = false;
    source.reason = "compute_pipeline_capacity";
    return std::move(source);
  }
  if (source.source_text.size() != specialization.exact_source_bytes ||
      source.source_text.capacity() != frozen_capacity ||
      source.source_text.data() != frozen_storage ||
      !backend_source_recipe::string_external_storage_within(source.source_text,
                                                             storage_upper)) {
    source.ok = false;
    source.reason = "compute_pipeline_capacity";
    return std::move(source);
  }
  source.source_text_upper_bytes = specialization.source_upper_bytes;
  source.metadata = {};
  source.canonical_ir_bytes = {};
  source.ok = true;
  source.reason = "ok";
  return std::move(source);
}

} // namespace rund::node::accel::detail
