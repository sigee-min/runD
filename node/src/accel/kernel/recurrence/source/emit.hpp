#pragma once

#include "../../backend/source/sink.hpp"
#include "../../recurrence.hpp"

#include <kernel/program/compute/lowering/layout.hpp>
#include <kernel/program/compute/lowering/metal/syntax.hpp>
#include <kernel/program/compute/lowering/vulkan/syntax.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rund::node::accel::detail::recurrence_source_detail {

using rund::kernel::ArtifactKey;
using rund::kernel::ComputeApi;
using rund::kernel::ComputeScalar;

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

[[nodiscard]] inline ArtifactKey RecurrenceKey(ArtifactKey source,
                                                const bool history) noexcept {
  ArtifactKey key = source;
  // Variant is an orthogonal executable-identity dimension. Canonical graph
  // and operation hashes remain the sole semantic graph identity.
  key.variant = history
                    ? rund::kernel::LoweringArtifactVariant::HistoryRecurrence
                    : rund::kernel::LoweringArtifactVariant::Recurrence;
  return key;
}

template <typename Sink>
[[nodiscard]] bool AppendSafeIdentifier(
    Sink &sink, const std::string_view name) noexcept(
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
[[nodiscard]] bool AppendVariable(
    Sink &sink, const std::string_view prefix,
    const std::size_t index) noexcept(noexcept(sink.append(std::string_view{}))) {
  return sink.append(prefix) &&
         backend_source_recipe::append_decimal(sink, index);
}

template <typename Sink>
[[nodiscard]] bool AppendBindingSymbol(
    Sink &sink, const std::string_view access,
    const std::string_view name) noexcept(noexcept(sink.append(std::string_view{}))) {
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
                                     const std::size_t index) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
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
          input.load_end > output.value_end || nested_count == nested.size()) {
        return false;
      }
      nested[nested_count++] = &input;
    }
    std::sort(
        nested.begin(), nested.begin() + nested_count,
        [](const SourceBinding *left, const SourceBinding *right) noexcept {
          return left->load_begin < right->load_begin;
        });
    std::size_t cursor = output.value_begin;
    for (std::size_t index = 0u; index < nested_count; ++index) {
      const SourceBinding &input = *nested[index];
      const std::size_t ordinal =
          static_cast<std::size_t>(&input - inputs.data());
      if (input.load_begin < cursor ||
          !sink.append(source.substr(cursor, input.load_begin - cursor)) ||
          !append_variable(
              sink, ordinal < output_count ? "rund_carry_" : "rund_invariant_",
              ordinal)) {
        return false;
      }
      cursor = input.load_end;
    }
    return sink.append(source.substr(cursor, output.value_end - cursor));
  }

  template <typename Sink>
  [[nodiscard]] bool append_prelude(Sink &sink) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
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
          !append_variable(
              sink, index < output_count ? "rund_carry_" : "rund_invariant_",
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
          !sink.append(" ") || !append_variable(sink, "rund_next_", index) ||
          !sink.append(" = ") || !append_variable(sink, "rund_carry_", index) ||
          !sink.append(";\n")) {
        return false;
      }
    }
    return sink.append("  for (uint rund_iteration = 0u; rund_iteration < ") &&
           sink.append(api == ComputeApi::Metal ? "rund_iterations"
                                                : "rund_dispatch.iterations") &&
           sink.append("; ++rund_iteration) {\n");
  }

  template <typename Sink>
  [[nodiscard]] bool append_store(Sink &sink, const OutputBinding &output,
                                  const std::size_t index,
                                  const bool history_store) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
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
    if (history_store && (!sink.append(" + rund_iteration * ") ||
                          !backend_source_recipe::append_decimal(
                              sink, output.history_pitch_bytes) ||
                          !sink.append("u"))) {
      return false;
    }
    return sink.append(" + gid * RundStride_") &&
           AppendBindingSymbol(sink, "write_", output.name) &&
           sink.append(", ") &&
           append_variable(sink, history_store ? "rund_next_" : "rund_carry_",
                           index) &&
           sink.append(");\n");
  }

  template <typename Sink>
  [[nodiscard]] bool append_epilogue(Sink &sink) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
    if (history) {
      for (std::size_t index = 0u; index < output_count; ++index) {
        if (!append_store(sink, outputs[index], index, true)) {
          return false;
        }
      }
    }
    for (std::size_t index = 0u; index < output_count; ++index) {
      if (!sink.append("    ") ||
          !append_variable(sink, "rund_carry_", index) || !sink.append(" = ") ||
          !append_variable(sink, "rund_next_", index) || !sink.append(";\n")) {
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
          sink, event.index < output_count ? "rund_carry_" : "rund_invariant_",
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
  [[nodiscard]] bool operator()(Sink &sink) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
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

} // namespace rund::node::accel::detail::recurrence_source_detail
