#pragma once

#include "../../kernel/backend/exception.hpp"
#include "../../kernel/backend/source_recipe.hpp"
#include "../../kernel/preparation.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace rund::node::accel::detail {

// Metal exposes 31 buffer-table entries (0...30). Pipeline-private execution
// owns the final entry for device-side suppression, leaving 30 application
// entries. Standalone kernels do not use this transformed ABI.
inline constexpr std::size_t kMetalPipelineApplicationArgumentCapacity = 30u;
inline constexpr std::size_t kMetalPipelineGuardBinding =
    kMetalPipelineApplicationArgumentCapacity;
inline constexpr std::size_t kMetalPipelineArgumentCapacity =
    kMetalPipelineGuardBinding + 1u;

namespace metal_pipeline_guard_detail {

[[nodiscard]] inline bool Identifier(const char value) noexcept {
  const auto byte = static_cast<unsigned char>(value);
  return std::isalnum(byte) != 0 || value == '_';
}

[[nodiscard]] inline std::size_t SkipSpace(const std::string_view source,
                                           std::size_t offset) noexcept {
  while (offset < source.size() &&
         std::isspace(static_cast<unsigned char>(source[offset])) != 0) {
    ++offset;
  }
  return offset;
}

[[nodiscard]] inline std::size_t
MatchingParen(const std::string_view source, const std::size_t open) noexcept {
  if (open >= source.size() || source[open] != '(') {
    return std::string_view::npos;
  }
  std::size_t depth = 0u;
  for (std::size_t index = open; index < source.size(); ++index) {
    if (source[index] == '(') {
      ++depth;
    } else if (source[index] == ')' && --depth == 0u) {
      return index;
    }
  }
  return std::string_view::npos;
}

struct Entry final {
  std::size_t arguments_begin{};
  std::size_t arguments{};
  std::size_t body{};
};

inline constexpr std::size_t kMaxPipelineSourceEntryCount = 10u;
using EntryList = std::array<Entry, kMaxPipelineSourceEntryCount>;

[[nodiscard]] inline bool Entries(const std::string_view source,
                                  EntryList &entries,
                                  std::size_t &entry_count) noexcept {
  entry_count = 0u;
  constexpr std::string_view marker = "kernel void";
  std::size_t search = 0u;
  while (true) {
    const std::size_t at = source.find(marker, search);
    if (at == std::string_view::npos) {
      return entry_count != 0u;
    }
    const std::size_t after = at + marker.size();
    if ((at != 0u && Identifier(source[at - 1u])) ||
        (after < source.size() && Identifier(source[after]))) {
      search = after;
      continue;
    }
    std::size_t open = source.find('(', after);
    if (open == std::string_view::npos) {
      return false;
    }
    std::size_t close = MatchingParen(source, open);
    if (close == std::string_view::npos) {
      return false;
    }
    // Some generated entry names are macro calls, for example
    // RUND_KERNEL(name)(arguments). Walk through those name parentheses until
    // the actual function-argument list has been consumed.
    std::size_t next = SkipSpace(source, close + 1u);
    while (next < source.size() && source[next] == '(') {
      open = next;
      close = MatchingParen(source, open);
      if (close == std::string_view::npos) {
        return false;
      }
      next = SkipSpace(source, close + 1u);
    }
    const std::size_t body = source.find('{', close + 1u);
    const std::size_t next_kernel = source.find(marker, close + 1u);
    if (body == std::string_view::npos ||
        (next_kernel != std::string_view::npos && next_kernel < body)) {
      return false;
    }
    if (entry_count == entries.size()) {
      return false;
    }
    entries[entry_count++] = Entry{
        .arguments_begin = open + 1u,
        .arguments = close,
        .body = body,
    };
    search = body + 1u;
  }
}

} // namespace metal_pipeline_guard_detail

// Allocation-free source-capacity authority for the Pipeline-private ABI.
// The public planner applies it to canonical Map text; runtime applies it to
// the specialized/controlled text. Specialization never changes the number of
// kernel entries, so both paths reserve the same final source envelope.
[[nodiscard]] inline bool PipelinePrivateMetalSourceUpperBytes(
    const std::uint64_t source_upper, const std::uint64_t entry_count,
    const bool pipeline_private, std::uint64_t &upper) noexcept {
  upper = source_upper;
  if (!pipeline_private) {
    return true;
  }
  constexpr std::string_view argument =
      ", device const uint *rund_pipeline_guard [[buffer(30)]]";
  constexpr std::string_view check =
      " if (rund_pipeline_guard[0] != 0u) { return; }";
  std::uint64_t growth = 0u;
  return entry_count != 0u &&
         rund::kernel::checked::mul(
             entry_count,
             static_cast<std::uint64_t>(argument.size() + check.size()),
             growth) &&
         rund::kernel::checked::add(upper, growth, upper);
}

// Pipeline-private Metal functions carry one uniform pointer to a uint guard.
// Unowned/control commands bind a frozen zero word. A recurrence-owned command
// binds directly to ResidentState::stopped, so every retained payload command
// can remain in one immutable ICB and self-suppress on the device.
[[nodiscard]] inline std::string
PipelinePrivateMetalSource(std::string source,
                           const std::uint64_t reserved_upper = 0u) {
  std::uint64_t input_storage_upper = 0u;
  if (!backend_source_recipe::string_external_storage_upper_bytes(
          std::max<std::uint64_t>(source.size(), reserved_upper),
          input_storage_upper) ||
      !backend_source_recipe::string_external_storage_within(
          source, input_storage_upper)) {
    return {};
  }
  if (!IsPipelinePrivatePreparation(CurrentKernelPreparationMode())) {
    return source;
  }
  constexpr std::string_view reserved = "buffer(30)";
  if (source.find(reserved) != std::string::npos) {
    return {};
  }
  metal_pipeline_guard_detail::EntryList entries{};
  std::size_t entry_count = 0u;
  try {
    if (!metal_pipeline_guard_detail::Entries(source, entries, entry_count)) {
      return {};
    }
    std::uint64_t required_upper = 0u;
    if (!PipelinePrivateMetalSourceUpperBytes(source.size(), entry_count, true,
                                              required_upper)) {
      return {};
    }
    const std::uint64_t final_upper = std::max(required_upper, reserved_upper);
    if (final_upper > std::numeric_limits<std::size_t>::max()) {
      return {};
    }
    if (!backend_source_recipe::reserve_string(source, final_upper)) {
      return {};
    }
    std::uint64_t final_storage_upper = 0u;
    if (!backend_source_recipe::string_external_storage_upper_bytes(
            final_upper, final_storage_upper) ||
        !backend_source_recipe::string_external_storage_within(
            source, final_storage_upper)) {
      return {};
    }
    const std::size_t frozen_capacity = source.capacity();
    constexpr std::string_view first_argument =
        "device const uint *rund_pipeline_guard [[buffer(30)]]";
    constexpr std::string_view next_argument =
        ", device const uint *rund_pipeline_guard [[buffer(30)]]";
    constexpr std::string_view check =
        " if (rund_pipeline_guard[0] != 0u) { return; }";
    for (std::size_t reverse = entry_count; reverse != 0u; --reverse) {
      const metal_pipeline_guard_detail::Entry &entry = entries[reverse - 1u];
      source.insert(entry.body + 1u, check);
      const std::size_t first =
          metal_pipeline_guard_detail::SkipSpace(source, entry.arguments_begin);
      source.insert(entry.arguments,
                    first == entry.arguments ? first_argument : next_argument);
    }
    if (source.size() > final_upper || source.capacity() != frozen_capacity ||
        !backend_source_recipe::string_external_storage_within(
            source, final_storage_upper)) {
      return {};
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return {};
  }
  return source;
}

[[nodiscard]] inline std::string
MetalPipelineCacheKey(const std::string_view key) {
  if (!IsPipelinePrivatePreparation(CurrentKernelPreparationMode())) {
    return std::string{key};
  }
  std::string scoped{"pipeline-private:"};
  scoped.append(key);
  return scoped;
}

} // namespace rund::node::accel::detail
