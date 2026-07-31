#pragma once

#include "../../kernel/preparation.hpp"

#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

[[nodiscard]] inline bool Entries(const std::string_view source,
                                  std::vector<Entry> &entries) {
  constexpr std::string_view marker = "kernel void";
  std::size_t search = 0u;
  while (true) {
    const std::size_t at = source.find(marker, search);
    if (at == std::string_view::npos) {
      return !entries.empty();
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
    entries.push_back(Entry{
        .arguments_begin = open + 1u,
        .arguments = close,
        .body = body,
    });
    search = body + 1u;
  }
}

} // namespace metal_pipeline_guard_detail

// Pipeline-private Metal functions carry one uniform pointer to a uint guard.
// Unowned/control commands bind a frozen zero word. A recurrence-owned command
// binds directly to ResidentState::stopped, so every retained payload command
// can remain in one immutable ICB and self-suppress on the device.
[[nodiscard]] inline std::string
PipelinePrivateMetalSource(std::string source) {
  if (!IsPipelinePrivatePreparation(CurrentKernelPreparationMode())) {
    return source;
  }
  constexpr std::string_view reserved = "buffer(30)";
  if (source.find(reserved) != std::string::npos) {
    return {};
  }
  std::vector<metal_pipeline_guard_detail::Entry> entries;
  try {
    if (!metal_pipeline_guard_detail::Entries(source, entries)) {
      return {};
    }
    constexpr std::string_view first_argument =
        "device const uint *rund_pipeline_guard [[buffer(30)]]";
    constexpr std::string_view next_argument =
        ", device const uint *rund_pipeline_guard [[buffer(30)]]";
    constexpr std::string_view check =
        " if (rund_pipeline_guard[0] != 0u) { return; }";
    for (auto entry = entries.rbegin(); entry != entries.rend(); ++entry) {
      source.insert(entry->body + 1u, check);
      const std::size_t first = metal_pipeline_guard_detail::SkipSpace(
          source, entry->arguments_begin);
      source.insert(entry->arguments,
                    first == entry->arguments ? first_argument : next_argument);
    }
  } catch (...) {
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
