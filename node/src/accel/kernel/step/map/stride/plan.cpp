#include "local.hpp"

#include "../../../backend/source/marker.hpp"
#include "../../../backend/source/edit.hpp"
#include "../../../backend/source/storage.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] bool FindMapDeclarationValue(const std::string_view source,
                                           const std::string_view qualifier,
                                           const std::string_view symbol_prefix,
                                           const std::string_view access_prefix,
                                           const std::string_view binding_name,
                                           std::size_t &value_at) noexcept {
  value_at = std::string_view::npos;
  std::size_t search = 0u;
  while (search < source.size()) {
    const std::size_t at = source.find(qualifier, search);
    if (at == std::string_view::npos) {
      break;
    }
    std::size_t cursor = at;
    if (backend_source_marker::consume_fragment(source, cursor, qualifier) &&
        backend_source_marker::consume_fragment(source, cursor,
                                                symbol_prefix) &&
        backend_source_marker::consume_fragment(source, cursor,
                                                access_prefix) &&
        backend_source_marker::consume_safe_identifier(source, cursor,
                                                       binding_name) &&
        backend_source_marker::consume_fragment(source, cursor, " = ")) {
      if (value_at != std::string_view::npos) {
        return false;
      }
      value_at = cursor;
    }
    search = at + 1u;
  }
  return value_at != std::string_view::npos;
}

[[nodiscard]] bool PlanMapDeclarationValue(
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

[[nodiscard]] bool FindMetalMapPointeeToken(const std::string_view source,
                                            const bool is_read,
                                            const std::string_view binding_name,
                                            const std::uint64_t binding_buffer,
                                            std::size_t &begin,
                                            std::size_t &end) noexcept {
  const std::string_view qualifier = is_read
                                         ? std::string_view{"    const device "}
                                         : std::string_view{"    device "};
  const std::string_view access =
      is_read ? std::string_view{"read_"} : std::string_view{"write_"};
  begin = std::string_view::npos;
  end = std::string_view::npos;
  std::array<char, 20u> binding_buffer_storage{};
  const std::string_view binding_buffer_text =
      backend_source_recipe::decimal_characters(binding_buffer,
                                                binding_buffer_storage);
  if (binding_buffer_text.empty()) {
    return false;
  }
  std::size_t search = 0u;
  while (search < source.size()) {
    const std::size_t at = source.find(qualifier, search);
    if (at == std::string_view::npos) {
      break;
    }
    std::size_t cursor = at;
    const bool prefix =
        backend_source_marker::consume_fragment(source, cursor, qualifier);
    const std::size_t token_begin = cursor;
    if ((at == 0u || source[at - 1u] == '\n') && prefix &&
        backend_source_marker::consume_fragment(source, cursor, "uchar* ") &&
        backend_source_marker::consume_fragment(source, cursor, access) &&
        backend_source_marker::consume_safe_identifier(source, cursor,
                                                       binding_name) &&
        backend_source_marker::consume_fragment(source, cursor, " [[buffer(") &&
        backend_source_marker::consume_fragment(source, cursor,
                                                binding_buffer_text) &&
        backend_source_marker::consume_fragment(source, cursor, ")]],\n")) {
      if (begin != std::string_view::npos) {
        return false;
      }
      begin = token_begin;
      end = token_begin + std::string_view{"uchar"}.size();
    }
    search = at + 1u;
  }
  return begin != std::string_view::npos;
}

[[nodiscard]] bool
PlanMetalMapWordPointee(const std::span<MapSourceEdit> edits,
                        std::size_t &edit_count, const std::string_view source,
                        const bool is_read, const std::string_view binding_name,
                        const std::uint64_t binding_buffer) noexcept {
  if (edit_count == edits.size()) {
    return false;
  }
  MapSourceEdit candidate{};
  if (!FindMetalMapPointeeToken(source, is_read, binding_name, binding_buffer,
                                candidate.begin, candidate.end)) {
    return false;
  }
  constexpr std::string_view replacement = "uint";
  std::copy(replacement.begin(), replacement.end(),
            candidate.replacement.begin());
  candidate.replacement_size = static_cast<std::uint8_t>(replacement.size());
  edits[edit_count++] = candidate;
  return true;
}

[[nodiscard]] MapSourceSpecialization
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
    const std::uint64_t binding_buffer =
        is_read ? read + 1u : plan.input_buffer_count + write + 1u;
    read += is_read ? 1u : 0u;
    write += is_write ? 1u : 0u;
    if (ref == nullptr || ref->element_bytes != element_bytes ||
        ref->stride_bytes < element_bytes) {
      result.reason = "compute_resident_stride_invalid";
      return result;
    }
    if (source.key.api == rund::kernel::ComputeApi::Metal &&
        MetalMapBindingWordAligned(*ref) &&
        !PlanMetalMapWordPointee(std::span<MapSourceEdit>{result.edits},
                                 result.edit_count, source.source_text, is_read,
                                 metadata.binding_names[index],
                                 binding_buffer)) {
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

} // namespace rund::node::accel::detail
