#pragma once

#include <cstddef>
#include <limits>
#include <span>

namespace rund::compute::detail {

[[nodiscard]] inline std::size_t
output_count(const std::span<const std::size_t> aliases,
             const std::size_t physical) noexcept {
  return aliases.empty() ? physical : aliases.size();
}

[[nodiscard]] inline std::size_t
output_index(const std::span<const std::size_t> aliases,
             const std::size_t logical) noexcept {
  if (aliases.empty()) {
    return logical;
  }
  return logical < aliases.size() ? aliases[logical]
                                  : std::numeric_limits<std::size_t>::max();
}

} // namespace rund::compute::detail
