#pragma once

#include <cstddef>
#include <string_view>

namespace program_compute_contract::lowering_support {

[[nodiscard]] inline std::size_t CountOccurrences(
    const std::string_view haystack,
    const std::string_view needle) noexcept {
  if (needle.empty()) {
    return 0u;
  }
  std::size_t count = 0u;
  std::size_t offset = 0u;
  while ((offset = haystack.find(needle, offset)) != std::string_view::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

}  // namespace program_compute_contract::lowering_support
