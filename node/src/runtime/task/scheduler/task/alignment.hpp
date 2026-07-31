#pragma once

#include <cstddef>
#include <limits>

namespace rund::node::alignment {

[[nodiscard]] constexpr bool power(const std::size_t value) noexcept {
  return value != 0u && (value & (value - 1u)) == 0u;
}

[[nodiscard]] constexpr bool up(const std::size_t value,
                                const std::size_t boundary,
                                std::size_t &result) noexcept {
  if (!power(boundary)) {
    return false;
  }
  const std::size_t mask = boundary - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

} // namespace rund::node::alignment
