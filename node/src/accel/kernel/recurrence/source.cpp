#include "source.hpp"

#include "../backend/source_recipe.hpp"

#include <kernel/program/compute/lowering/layout.hpp>
#include <kernel/program/compute/lowering/metal/syntax.hpp>
#include <kernel/program/compute/lowering/text.hpp>
#include <kernel/program/compute/lowering/vulkan/syntax.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace rund::node::accel::detail {
using rund::kernel::ArtifactKey;
using rund::kernel::ComputeApi;
using rund::kernel::ComputeBindingAccess;
using rund::kernel::ComputeScalar;
using rund::kernel::ExecutionMetadata;
using rund::kernel::LoweringArtifact;

namespace {

inline constexpr std::size_t RecurrenceBindingCapacity =
    static_cast<std::size_t>(rund::kernel::kMaxComputeBindingCount);

struct SourceBinding final {
  std::string_view name{};
  std::size_t load_begin{};
  std::size_t load_end{};
  std::uint64_t element_bytes{};
  bool uniform{};
};

struct OutputBinding final {
  std::string_view name{};
  std::size_t store_begin{};
  std::size_t value_begin{};
  std::size_t value_end{};
  std::size_t store_end{};
  std::uint64_t element_bytes{};
  std::uint64_t history_pitch_bytes{};
};

enum class SourceEventKind : std::uint8_t {
  Variant,
  MetalName,
  Body,
  Input,
  Output,
  Epilogue,
};

struct SourceEvent final {
  std::size_t begin{};
  std::size_t end{};
  std::uint8_t index{};
  SourceEventKind kind{SourceEventKind::Variant};
};

inline constexpr std::size_t RecurrenceSourceEventCapacity =
    2u * RecurrenceBindingCapacity + 4u;

[[nodiscard]] bool FindOne(const std::string_view source,
                           const std::string_view needle,
                           std::size_t &at) noexcept {
  if (needle.empty()) {
    return false;
  }
  at = source.find(needle);
  if (at == std::string::npos ||
      source.find(needle, at + needle.size()) != std::string::npos) {
    return false;
  }
  return true;
}

[[nodiscard]] ArtifactKey RecurrenceKey(ArtifactKey source,
                                        const bool history) noexcept {
  ArtifactKey key = source;
  // Variant is an orthogonal executable-identity dimension. Canonical graph
  // and operation hashes remain the sole semantic graph identity.
  key.variant = history
                    ? rund::kernel::LoweringArtifactVariant::HistoryRecurrence
                    : rund::kernel::LoweringArtifactVariant::Recurrence;
  return key;
}

[[nodiscard]] bool Consume(const std::string_view source, std::size_t &cursor,
                           const std::string_view fragment) noexcept {
  if (cursor > source.size() || fragment.size() > source.size() - cursor ||
      source.compare(cursor, fragment.size(), fragment) != 0) {
    return false;
  }
  cursor += fragment.size();
  return true;
}

[[nodiscard]] bool ConsumeSafeIdentifier(const std::string_view source,
                                         std::size_t &cursor,
                                         const std::string_view name) noexcept {
  if (name.empty()) {
    return Consume(source, cursor, "empty");
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

[[nodiscard]] bool ConsumeSymbol(const std::string_view source,
                                 std::size_t &cursor,
                                 const std::string_view access,
                                 const std::string_view name) noexcept {
  return Consume(source, cursor, access) &&
         ConsumeSafeIdentifier(source, cursor, name);
}

template <typename Match>
[[nodiscard]] bool FindUniqueStructured(const std::string_view source,
                                        const std::string_view anchor,
                                        Match &&match, std::size_t &begin,
                                        std::size_t &end) noexcept {
  begin = std::string_view::npos;
  end = std::string_view::npos;
  std::size_t search = 0u;
  while (search < source.size()) {
    const std::size_t at = source.find(anchor, search);
    if (at == std::string_view::npos) {
      break;
    }
    std::size_t cursor = at;
    if (match(cursor)) {
      if (begin != std::string_view::npos) {
        return false;
      }
      begin = at;
      end = cursor;
    }
    search = at + 1u;
  }
  return begin != std::string_view::npos;
}

[[nodiscard]] bool FindInputLoad(const std::string_view source,
                                 const ComputeApi api,
                                 const ComputeScalar scalar,
                                 const std::string_view name,
                                 const bool uniform, std::size_t &begin,
                                 std::size_t &end) noexcept {
  const std::string_view load =
      api == ComputeApi::Metal
          ? rund::kernel::compute_lowering_detail::MetalLoadFunction(scalar)
          : rund::kernel::compute_lowering_detail::VulkanLoadPrefix(scalar);
  return FindUniqueStructured(
      source, load,
      [&](std::size_t &cursor) noexcept {
        return Consume(source, cursor, load) &&
               (api == ComputeApi::Metal
                    ? Consume(source, cursor, "(")
                    : Consume(source, cursor, "_") &&
                          ConsumeSymbol(source, cursor, "read_", name) &&
                          Consume(source, cursor, "(")) &&
               (api != ComputeApi::Metal ||
                (ConsumeSymbol(source, cursor, "read_", name) &&
                 Consume(source, cursor, ", "))) &&
               Consume(source, cursor, "RundBase_") &&
               ConsumeSymbol(source, cursor, "read_", name) &&
               (uniform ||
                (Consume(source, cursor, " + gid * RundStride_") &&
                 ConsumeSymbol(source, cursor, "read_", name))) &&
               Consume(source, cursor, ")");
      },
      begin, end);
}

[[nodiscard]] bool FindOutputStore(
    const std::string_view source, const ComputeApi api,
    const ComputeScalar scalar, const std::string_view name,
    std::size_t &begin, std::size_t &value_begin, std::size_t &value_end,
    std::size_t &end) noexcept {
  const std::string_view store =
      api == ComputeApi::Metal
          ? rund::kernel::compute_lowering_detail::MetalStoreFunction(scalar)
          : rund::kernel::compute_lowering_detail::VulkanStorePrefix(scalar);
  std::size_t prefix_end = 0u;
  if (!FindUniqueStructured(
          source, "  ",
          [&](std::size_t &cursor) noexcept {
            return Consume(source, cursor, "  ") &&
                   Consume(source, cursor, store) &&
                   (api == ComputeApi::Metal
                        ? Consume(source, cursor, "(") &&
                              ConsumeSymbol(source, cursor, "write_", name) &&
                              Consume(source, cursor, ", ")
                        : Consume(source, cursor, "_") &&
                              ConsumeSymbol(source, cursor, "write_", name) &&
                              Consume(source, cursor, "(")) &&
                   Consume(source, cursor, "RundBase_") &&
                   ConsumeSymbol(source, cursor, "write_", name) &&
                   Consume(source, cursor, " + gid * RundStride_") &&
                   ConsumeSymbol(source, cursor, "write_", name) &&
                   Consume(source, cursor, ", ");
          },
          begin, prefix_end)) {
    return false;
  }
  value_begin = prefix_end;
  value_end = source.find(");\n", value_begin);
  if (value_end == std::string_view::npos) {
    return false;
  }
  end = value_end + 3u;
  return true;
}

[[nodiscard]] bool SourceBindings(
    const ExecutionMetadata &metadata, const std::string_view source,
    const ComputeApi api, const ComputeScalar scalar,
    const std::span<const std::uint64_t> history_pitch_bytes,
    std::array<SourceBinding, RecurrenceBindingCapacity> &inputs,
    std::size_t &input_count,
    std::array<OutputBinding, RecurrenceBindingCapacity> &outputs,
    std::size_t &output_count) noexcept {
  if (!metadata.ok ||
      metadata.binding_accesses.size() != metadata.binding_names.size() ||
      metadata.input_element_bytes.size() != metadata.read_count ||
      metadata.output_element_bytes.size() != metadata.write_count ||
      metadata.binding_accesses.size() > RecurrenceBindingCapacity ||
      metadata.read_count > inputs.size() || metadata.write_count > outputs.size()) {
    return false;
  }
  input_count = 0u;
  output_count = 0u;
  for (std::size_t index = 0u; index < metadata.binding_accesses.size();
       ++index) {
    const ComputeBindingAccess access = metadata.binding_accesses[index];
    if (access == ComputeBindingAccess::Read) {
      if (input_count >= metadata.input_element_bytes.size() ||
          input_count >= 64u) {
        return false;
      }
      const std::uint64_t bit = std::uint64_t{1u} << input_count;
      const bool direct = (metadata.direct_read_mask & bit) != 0u;
      const bool uniform = (metadata.uniform_read_mask & bit) != 0u;
      if (direct == uniform) {
        return false;
      }
      SourceBinding binding{
          .name = metadata.binding_names[index],
          .element_bytes = metadata.input_element_bytes[input_count],
          .uniform = uniform,
      };
      if (!FindInputLoad(source, api, scalar, binding.name, binding.uniform,
                         binding.load_begin, binding.load_end)) {
        return false;
      }
      inputs[input_count++] = binding;
    } else if (access == ComputeBindingAccess::Write) {
      if (output_count >= metadata.output_element_bytes.size()) {
        return false;
      }
      OutputBinding binding{
          .name = metadata.binding_names[index],
          .element_bytes = metadata.output_element_bytes[output_count],
          .history_pitch_bytes = history_pitch_bytes.empty()
                                     ? 0u
                                     : history_pitch_bytes[output_count],
      };
      if (!FindOutputStore(source, api, scalar, binding.name,
                           binding.store_begin, binding.value_begin,
                           binding.value_end, binding.store_end)) {
        return false;
      }
      outputs[output_count++] = binding;
    } else {
      return false;
    }
  }
  return input_count == metadata.input_element_bytes.size() &&
         output_count == metadata.output_element_bytes.size();
}

template <typename Sink>
[[nodiscard]] bool AppendSafeIdentifier(Sink &sink,
                                        const std::string_view name) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  if (name.empty()) {
    return sink.append("empty");
  }
  std::array<char, 2u> digits{};
  for (const char value : name) {
    const auto byte =
        static_cast<rund::kernel::u8>(static_cast<unsigned char>(value));
    digits[0] = rund::kernel::compute_lowering_detail::HexDigit(
        static_cast<rund::kernel::u8>((byte >> 4u) & 0x0fu));
    digits[1] = rund::kernel::compute_lowering_detail::HexDigit(
        static_cast<rund::kernel::u8>(byte & 0x0fu));
    if (!sink.append(std::string_view{digits.data(), digits.size()})) {
      return false;
    }
  }
  return true;
}

template <typename Sink>
[[nodiscard]] bool AppendVariable(Sink &sink, const std::string_view prefix,
                                  const std::size_t index) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return sink.append(prefix) && backend_source_recipe::append_decimal(sink, index);
}

template <typename Sink>
[[nodiscard]] bool AppendBindingSymbol(
    Sink &sink, const std::string_view access,
    const std::string_view name) noexcept(noexcept(sink.append(
    std::string_view{}))) {
  return sink.append(access) && AppendSafeIdentifier(sink, name);
}

struct RecurrenceSourceRecipe final {
  std::string_view source{};
  ArtifactKey before{};
  ArtifactKey after{};
  ComputeScalar scalar{ComputeScalar::Lane32};
  ComputeApi api{ComputeApi::Metal};
  std::array<SourceBinding, RecurrenceBindingCapacity> inputs{};
  std::array<OutputBinding, RecurrenceBindingCapacity> outputs{};
  std::array<SourceEvent, RecurrenceSourceEventCapacity> events{};
  std::size_t input_count{};
  std::size_t output_count{};
  std::size_t event_count{};
  bool history{};
  bool ok{};

  template <typename Sink>
  [[nodiscard]] bool append_variable(Sink &sink, const std::string_view prefix,
                                     const std::size_t index) const noexcept(
      noexcept(sink.append(std::string_view{}))) {
    return AppendVariable(sink, prefix, index);
  }

  template <typename Sink>
  [[nodiscard]] bool append_original_value(Sink &sink,
                                           const OutputBinding &output) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
    std::array<const SourceBinding *, RecurrenceBindingCapacity> nested{};
    std::size_t nested_count = 0u;
    for (std::size_t index = 0u; index < input_count; ++index) {
      const SourceBinding &input = inputs[index];
      const bool overlaps = input.load_begin < output.value_end &&
                            input.load_end > output.value_begin;
      if (!overlaps) {
        continue;
      }
      if (input.load_begin < output.value_begin ||
          input.load_end > output.value_end ||
          nested_count == nested.size()) {
        return false;
      }
      nested[nested_count++] = &input;
    }
    std::sort(nested.begin(), nested.begin() + nested_count,
              [](const SourceBinding *left,
                 const SourceBinding *right) noexcept {
                return left->load_begin < right->load_begin;
              });
    std::size_t cursor = output.value_begin;
    for (std::size_t index = 0u; index < nested_count; ++index) {
      const SourceBinding &input = *nested[index];
      const std::size_t ordinal = static_cast<std::size_t>(&input - inputs.data());
      if (input.load_begin < cursor ||
          !sink.append(source.substr(cursor, input.load_begin - cursor)) ||
          !append_variable(sink,
                           ordinal < output_count ? "rund_carry_"
                                                  : "rund_invariant_",
                           ordinal)) {
        return false;
      }
      cursor = input.load_end;
    }
    return sink.append(source.substr(cursor, output.value_end - cursor));
  }

  template <typename Sink>
  [[nodiscard]] bool append_prelude(Sink &sink) const noexcept(
      noexcept(sink.append(std::string_view{}))) {
    const char *const scalar_type =
        api == ComputeApi::Metal
            ? rund::kernel::compute_lowering_detail::MetalType(scalar)
            : rund::kernel::compute_lowering_detail::VulkanType(scalar);
    if (scalar_type == nullptr) {
      return false;
    }
    for (std::size_t index = 0u; index < input_count; ++index) {
      if (!sink.append("  ") || !sink.append(scalar_type) ||
          !sink.append(" ") ||
          !append_variable(sink,
                           index < output_count ? "rund_carry_"
                                                : "rund_invariant_",
                           index) ||
          !sink.append(" = ") ||
          !sink.append(source.substr(inputs[index].load_begin,
                                     inputs[index].load_end -
                                         inputs[index].load_begin)) ||
          !sink.append(";\n")) {
        return false;
      }
    }
    for (std::size_t index = 0u; index < output_count; ++index) {
      if (!sink.append("  ") || !sink.append(scalar_type) ||
          !sink.append(" ") ||
          !append_variable(sink, "rund_next_", index) ||
          !sink.append(" = ") ||
          !append_variable(sink, "rund_carry_", index) ||
          !sink.append(";\n")) {
        return false;
      }
    }
    return sink.append("  for (uint rund_iteration = 0u; rund_iteration < ") &&
           sink.append(api == ComputeApi::Metal
                           ? "rund_iterations"
                           : "rund_dispatch.iterations") &&
           sink.append("; ++rund_iteration) {\n");
  }

  template <typename Sink>
  [[nodiscard]] bool append_store(Sink &sink, const OutputBinding &output,
                                  const std::size_t index,
                                  const bool history_store) const noexcept(
      noexcept(sink.append(std::string_view{}))) {
    const char *const store =
        api == ComputeApi::Metal
            ? rund::kernel::compute_lowering_detail::MetalStoreFunction(scalar)
            : rund::kernel::compute_lowering_detail::VulkanStorePrefix(scalar);
    if (store == nullptr || !sink.append(history_store ? "    " : "  ") ||
        !sink.append(store) ||
        (api == ComputeApi::Vulkan && !sink.append("_")) ||
        (api == ComputeApi::Metal && !sink.append("(")) ||
        !AppendBindingSymbol(sink, "write_", output.name) ||
        (api == ComputeApi::Metal && !sink.append(", ")) ||
        (api == ComputeApi::Vulkan && !sink.append("(")) ||
        !sink.append("RundBase_") ||
        !AppendBindingSymbol(sink, "write_", output.name)) {
      return false;
    }
    if (history_store &&
        (!sink.append(" + rund_iteration * ") ||
         !backend_source_recipe::append_decimal(
             sink, output.history_pitch_bytes) ||
         !sink.append("u"))) {
      return false;
    }
    return sink.append(" + gid * RundStride_") &&
           AppendBindingSymbol(sink, "write_", output.name) &&
           sink.append(", ") &&
           append_variable(sink,
                           history_store ? "rund_next_" : "rund_carry_",
                           index) &&
           sink.append(");\n");
  }

  template <typename Sink>
  [[nodiscard]] bool append_epilogue(Sink &sink) const noexcept(
      noexcept(sink.append(std::string_view{}))) {
    if (history) {
      for (std::size_t index = 0u; index < output_count; ++index) {
        if (!append_store(sink, outputs[index], index, true)) {
          return false;
        }
      }
    }
    for (std::size_t index = 0u; index < output_count; ++index) {
      if (!sink.append("    ") ||
          !append_variable(sink, "rund_carry_", index) ||
          !sink.append(" = ") ||
          !append_variable(sink, "rund_next_", index) ||
          !sink.append(";\n")) {
        return false;
      }
    }
    if (!sink.append("  }\n")) {
      return false;
    }
    if (!history) {
      for (std::size_t index = 0u; index < output_count; ++index) {
        if (!append_store(sink, outputs[index], index, false)) {
          return false;
        }
      }
    }
    return true;
  }

  template <typename Sink>
  [[nodiscard]] bool append_event(Sink &sink, const SourceEvent &event) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
    switch (event.kind) {
    case SourceEventKind::Variant:
      return sink.append(history ? "// artifact_variant=history_recurrence"
                                 : "// artifact_variant=recurrence");
    case SourceEventKind::MetalName:
      return sink.append("rund_compute_map_") &&
             backend_source_recipe::append_hex64_digits(sink,
                                                        after.op_hash_hi) &&
             sink.append("_") &&
             backend_source_recipe::append_hex64_digits(sink,
                                                        after.op_hash_lo) &&
             sink.append(history ? "_history_recurrence" : "_recurrence");
    case SourceEventKind::Body:
      if (api == ComputeApi::Metal &&
          (!sink.append("    constant uint& rund_iterations [[buffer(") ||
           !backend_source_recipe::append_decimal(
               sink, input_count + output_count + 1u) ||
           !sink.append(")]],\n") ||
           !sink.append(source.substr(event.begin, event.end - event.begin)))) {
        return false;
      }
      return append_prelude(sink);
    case SourceEventKind::Input:
      return append_variable(
          sink, event.index < output_count ? "rund_carry_"
                                           : "rund_invariant_",
          event.index);
    case SourceEventKind::Output:
      return sink.append("    ") &&
             append_variable(sink, "rund_next_", event.index) &&
             sink.append(" = ") &&
             append_original_value(sink, outputs[event.index]) &&
             sink.append(";\n");
    case SourceEventKind::Epilogue:
      return append_epilogue(sink);
    }
    return false;
  }

  template <typename Sink>
  [[nodiscard]] bool operator()(Sink &sink) const noexcept(
      noexcept(sink.append(std::string_view{}))) {
    if (!ok) {
      return false;
    }
    std::size_t cursor = 0u;
    for (std::size_t index = 0u; index < event_count; ++index) {
      const SourceEvent &event = events[index];
      if (event.begin < cursor || event.end < event.begin ||
          event.end > source.size() ||
          !sink.append(source.substr(cursor, event.begin - cursor)) ||
          !append_event(sink, event)) {
        return false;
      }
      cursor = event.end;
    }
    return sink.append(source.substr(cursor));
  }
};

[[nodiscard]] bool AddEvent(RecurrenceSourceRecipe &recipe,
                            const SourceEvent event) noexcept {
  if (recipe.event_count == recipe.events.size()) {
    return false;
  }
  recipe.events[recipe.event_count++] = event;
  return true;
}

[[nodiscard]] bool InputNestedInOutput(
    const SourceBinding &input,
    const std::span<const OutputBinding> outputs) noexcept {
  bool nested = false;
  for (const OutputBinding &output : outputs) {
    const bool overlaps = input.load_begin < output.store_end &&
                          input.load_end > output.store_begin;
    if (!overlaps) {
      continue;
    }
    if (nested || input.load_begin < output.value_begin ||
        input.load_end > output.value_end) {
      return false;
    }
    nested = true;
  }
  return nested;
}

[[nodiscard]] RecurrenceSourceRecipe BuildRecipe(
    const LoweringArtifact &artifact, const std::uint64_t expected_inputs,
    const std::uint64_t expected_outputs,
    const std::span<const std::uint64_t> history_pitch_bytes) noexcept {
  RecurrenceSourceRecipe recipe{};
  recipe.source = artifact.source_text;
  recipe.before = artifact.key;
  recipe.api = artifact.key.api;
  recipe.scalar = artifact.key.scalar;
  recipe.history = !history_pitch_bytes.empty();
  recipe.after = RecurrenceKey(recipe.before, recipe.history);
  const bool source_kind_matches =
      (recipe.api == ComputeApi::Metal &&
       artifact.kind == rund::kernel::LoweringArtifactKind::MetalSource) ||
      (recipe.api == ComputeApi::Vulkan &&
       artifact.kind == rund::kernel::LoweringArtifactKind::VulkanSource);
  if (!artifact.ok || !source_kind_matches || recipe.source.empty() ||
      artifact.source_text_upper_bytes < recipe.source.size() ||
      expected_outputs == 0u || expected_outputs > expected_inputs ||
      expected_inputs > RecurrenceBindingCapacity ||
      expected_outputs > RecurrenceBindingCapacity ||
      (recipe.history && history_pitch_bytes.size() != expected_outputs) ||
      recipe.before.variant !=
          rund::kernel::LoweringArtifactVariant::Canonical) {
    return recipe;
  }
  for (const std::uint64_t pitch : history_pitch_bytes) {
    if (pitch == 0u || pitch > std::numeric_limits<std::uint32_t>::max()) {
      return recipe;
    }
  }
  if (!SourceBindings(artifact.metadata, recipe.source, recipe.api,
                      recipe.scalar, history_pitch_bytes, recipe.inputs,
                      recipe.input_count, recipe.outputs,
                      recipe.output_count) ||
      recipe.input_count != expected_inputs ||
      recipe.output_count != expected_outputs) {
    return recipe;
  }
  const std::uint64_t scalar_bytes =
      rund::kernel::ComputeScalarBits(recipe.scalar) / 8u;
  for (std::size_t index = 0u; index < recipe.output_count; ++index) {
    if (recipe.inputs[index].uniform ||
        recipe.inputs[index].element_bytes != scalar_bytes ||
        recipe.outputs[index].element_bytes != scalar_bytes) {
      return recipe;
    }
  }

  std::size_t variant = 0u;
  if (!FindOne(recipe.source, "// artifact_variant=canonical", variant) ||
      !AddEvent(recipe, SourceEvent{
                            .begin = variant,
                            .end = variant +
                                   std::string_view{
                                       "// artifact_variant=canonical"}
                                       .size(),
                            .kind = SourceEventKind::Variant,
                        })) {
    return recipe;
  }
  if (recipe.api == ComputeApi::Metal) {
    std::size_t name_begin = 0u;
    std::size_t name_end = 0u;
    if (!FindUniqueStructured(
            recipe.source, "rund_compute_map_",
            [&](std::size_t &cursor) noexcept {
              std::array<char, 16u> hi{};
              std::array<char, 16u> lo{};
              backend_source_recipe::FixedBufferSink<16u> hi_sink{hi};
              backend_source_recipe::FixedBufferSink<16u> lo_sink{lo};
              return backend_source_recipe::append_hex64_digits(
                         hi_sink, recipe.before.op_hash_hi) &&
                     backend_source_recipe::append_hex64_digits(
                         lo_sink, recipe.before.op_hash_lo) &&
                     Consume(recipe.source, cursor, "rund_compute_map_") &&
                     Consume(recipe.source, cursor, hi_sink.text()) &&
                     Consume(recipe.source, cursor, "_") &&
                     Consume(recipe.source, cursor, lo_sink.text());
            },
            name_begin, name_end) ||
        !AddEvent(recipe, SourceEvent{
                              .begin = name_begin,
                              .end = name_end,
                              .kind = SourceEventKind::MetalName,
                          })) {
      return recipe;
    }
  }

  std::size_t body_begin = 0u;
  std::size_t body_end = 0u;
  if (recipe.api == ComputeApi::Metal) {
    constexpr std::string_view Gid =
        "    uint gid [[thread_position_in_grid]]) {\n";
    if (!FindOne(recipe.source, Gid, body_begin)) {
      return recipe;
    }
    body_end = body_begin + Gid.size();
  } else {
    constexpr std::string_view Dispatch =
        "layout(push_constant) uniform RundDispatch {\n"
        "  uint tile_count;\n"
        "  uint iterations;\n"
        "} rund_dispatch;\n";
    constexpr std::string_view Anchor =
        "  if (gid >= rund_dispatch.tile_count) { return; }\n";
    std::size_t declaration = 0u;
    if (!FindOne(recipe.source, Dispatch, declaration) ||
        !FindOne(recipe.source, Anchor, body_begin)) {
      return recipe;
    }
    body_begin += Anchor.size();
    body_end = body_begin;
  }
  if (!AddEvent(recipe, SourceEvent{
                            .begin = body_begin,
                            .end = body_end,
                            .kind = SourceEventKind::Body,
                        })) {
    return recipe;
  }
  const std::span<const OutputBinding> outputs{recipe.outputs.data(),
                                               recipe.output_count};
  for (std::size_t index = 0u; index < recipe.input_count; ++index) {
    const SourceBinding &input = recipe.inputs[index];
    const bool nested = InputNestedInOutput(input, outputs);
    if ((!nested &&
         !AddEvent(recipe, SourceEvent{
                               .begin = input.load_begin,
                               .end = input.load_end,
                               .index = static_cast<std::uint8_t>(index),
                               .kind = SourceEventKind::Input,
                           })) ||
        (nested && input.load_begin == input.load_end)) {
      return recipe;
    }
  }
  for (std::size_t index = 0u; index < recipe.output_count; ++index) {
    const OutputBinding &output = recipe.outputs[index];
    if (!AddEvent(recipe, SourceEvent{
                              .begin = output.store_begin,
                              .end = output.store_end,
                              .index = static_cast<std::uint8_t>(index),
                              .kind = SourceEventKind::Output,
                          })) {
      return recipe;
    }
  }
  const std::size_t epilogue = recipe.source.rfind("}\n");
  if (epilogue == std::string_view::npos || epilogue < body_end ||
      !AddEvent(recipe, SourceEvent{
                            .begin = epilogue,
                            .end = epilogue,
                            .kind = SourceEventKind::Epilogue,
                        })) {
    return recipe;
  }
  std::sort(recipe.events.begin(), recipe.events.begin() + recipe.event_count,
            [](const SourceEvent &left, const SourceEvent &right) noexcept {
              return left.begin < right.begin ||
                     (left.begin == right.begin && left.end < right.end);
            });
  std::size_t consumed = 0u;
  bool first = true;
  std::size_t prior = 0u;
  for (std::size_t index = 0u; index < recipe.event_count; ++index) {
    const SourceEvent &event = recipe.events[index];
    if (event.begin < consumed || event.end < event.begin ||
        event.end > recipe.source.size() ||
        (!first && event.begin == prior)) {
      return recipe;
    }
    consumed = event.end;
    prior = event.begin;
    first = false;
  }
  recipe.ok = true;
  return recipe;
}

[[nodiscard]] bool AddStorageEnvelope(std::uint64_t &bytes,
                                      const std::uint64_t count,
                                      const std::uint64_t width) noexcept {
  std::uint64_t payload = 0u;
  if (!rund::kernel::checked::mul(count, width, payload)) {
    return false;
  }
  if (payload != 0u &&
      !rund::kernel::checked::add(
          payload, backend_source_recipe::StringExternalStorageSlackBytes,
          payload)) {
    return false;
  }
  return rund::kernel::checked::add(bytes, payload, bytes);
}

[[nodiscard]] bool MetadataStorageUpperBytes(
    const ExecutionMetadata &metadata, std::uint64_t &bytes) noexcept {
  bytes = 0u;
  if (!AddStorageEnvelope(bytes, metadata.param_storage.size(),
                          sizeof(metadata.param_storage.front())) ||
      !AddStorageEnvelope(bytes, metadata.input_element_bytes.size(),
                          sizeof(metadata.input_element_bytes.front())) ||
      !AddStorageEnvelope(bytes, metadata.output_element_bytes.size(),
                          sizeof(metadata.output_element_bytes.front())) ||
      !AddStorageEnvelope(bytes, metadata.binding_accesses.size(),
                          sizeof(metadata.binding_accesses.front())) ||
      !AddStorageEnvelope(bytes, metadata.binding_names.size(),
                          sizeof(metadata.binding_names.front())) ||
      !AddStorageEnvelope(bytes, metadata.read_routes.size(),
                          sizeof(metadata.read_routes.front()))) {
    return false;
  }
  for (const std::string &name : metadata.binding_names) {
    std::uint64_t storage = 0u;
    if (!backend_source_recipe::string_external_storage_upper_bytes(
            name.size(), storage) ||
        !rund::kernel::checked::add(bytes, storage, bytes)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] MapRecurrenceSourcePlan PlanRecipe(
    const LoweringArtifact &artifact,
    const RecurrenceSourceRecipe &recipe) noexcept {
  MapRecurrenceSourcePlan plan{.history = recipe.history};
  if (!recipe.ok ||
      artifact.source_text_upper_bytes < artifact.source_text.size()) {
    return plan;
  }
  const auto count = [&](backend_source_recipe::CountSink &sink) noexcept {
    return recipe(sink);
  };
  const std::uint64_t inherited_upper_growth =
      artifact.source_text_upper_bytes - artifact.source_text.size();
  std::uint64_t metadata_storage = 0u;
  if (!backend_source_recipe::bytes(count, plan.exact_source_bytes) ||
      !rund::kernel::checked::add(plan.exact_source_bytes,
                                  inherited_upper_growth,
                                  plan.source_upper_bytes) ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          plan.source_upper_bytes, plan.source_storage_upper_bytes) ||
      !MetadataStorageUpperBytes(artifact.metadata, metadata_storage)) {
    plan.reason = "compute_pipeline_capacity";
    return plan;
  }
  plan.metadata_storage_upper_bytes = metadata_storage;
  plan.ok = true;
  plan.reason = "ok";
  return plan;
}

} // namespace

MapRecurrenceSourcePlan PlanMapRecurrenceSource(
    const LoweringArtifact &artifact, const std::uint64_t input_count,
    const std::uint64_t output_count,
    const std::span<const std::uint64_t> history_pitch_bytes) noexcept {
  const RecurrenceSourceRecipe recipe =
      BuildRecipe(artifact, input_count, output_count, history_pitch_bytes);
  return PlanRecipe(artifact, recipe);
}

bool MaterializeMapRecurrenceArtifact(
    const LoweringArtifact &canonical, const MapRecurrenceSourcePlan &plan,
    const std::uint64_t input_count, const std::uint64_t output_count,
    const std::span<const std::uint64_t> history_pitch_bytes,
    LoweringArtifact &artifact,
    const std::uint64_t source_reserve_upper_bytes) {
  const RecurrenceSourceRecipe recipe = BuildRecipe(
      canonical, input_count, output_count, history_pitch_bytes);
  const MapRecurrenceSourcePlan observed = PlanRecipe(canonical, recipe);
  if (!plan.ok || !observed.ok || !recipe.ok ||
      plan.exact_source_bytes != observed.exact_source_bytes ||
      plan.source_upper_bytes != observed.source_upper_bytes ||
      plan.source_storage_upper_bytes != observed.source_storage_upper_bytes ||
      plan.metadata_storage_upper_bytes !=
          observed.metadata_storage_upper_bytes ||
      plan.history != observed.history) {
    return false;
  }
  const std::uint64_t reserve_upper =
      std::max(plan.source_upper_bytes, source_reserve_upper_bytes);
  std::uint64_t reserve_storage_upper = 0u;
  std::uint64_t materialization_host_upper = 0u;
  if (!backend_source_recipe::string_external_storage_upper_bytes(
          reserve_upper, reserve_storage_upper) ||
      !rund::kernel::checked::add(
          plan.metadata_storage_upper_bytes, reserve_storage_upper,
          materialization_host_upper)) {
    return false;
  }
  try {
    LoweringArtifact transformed{
        .key = RecurrenceKey(canonical.key, plan.history),
        .kind = canonical.kind,
        .metadata = canonical.metadata,
        .source_text_upper_bytes = plan.source_upper_bytes,
        .ok = canonical.ok,
        .reason = canonical.reason,
    };
    transformed.source_text = backend_source_recipe::materialize(
        recipe, plan.exact_source_bytes, reserve_upper);
    if (transformed.source_text.empty() ||
        transformed.source_text.size() != plan.exact_source_bytes ||
        !backend_source_recipe::string_external_storage_within(
            transformed.source_text, reserve_storage_upper) ||
        transformed.retained_dynamic_memory_bytes() >
            materialization_host_upper) {
      return false;
    }
    transformed.metadata.map.op_hash_hi = transformed.key.op_hash_hi;
    transformed.metadata.map.op_hash_lo = transformed.key.op_hash_lo;
    artifact = std::move(transformed);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  } catch (const std::length_error &) {
    return false;
  }
}

[[nodiscard]] bool TransformSource(LoweringArtifact &artifact,
                                   const std::uint64_t input_count,
                                   const std::uint64_t output_count,
                                   const std::span<const std::uint64_t>
                                       history_pitch_bytes) {
  const MapRecurrenceSourcePlan plan = PlanMapRecurrenceSource(
      artifact, input_count, output_count, history_pitch_bytes);
  LoweringArtifact transformed{};
  if (!plan.ok || !MaterializeMapRecurrenceArtifact(
                      artifact, plan, input_count, output_count,
                      history_pitch_bytes, transformed)) {
    return false;
  }
  artifact = std::move(transformed);
  return true;
}
} // namespace rund::node::accel::detail
