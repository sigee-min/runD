#pragma once

#include <cstddef>
#include <limits>

namespace rund::compute::detail::size {

[[nodiscard]] constexpr bool add(const std::size_t left,
                                 const std::size_t right) noexcept {
  return left <= std::numeric_limits<std::size_t>::max() - right;
}

[[nodiscard]] constexpr bool add(const std::size_t left,
                                 const std::size_t right,
                                 std::size_t &value) noexcept {
  if (!add(left, right)) {
    return false;
  }
  value = left + right;
  return true;
}

[[nodiscard]] constexpr bool multiply(const std::size_t left,
                                      const std::size_t right) noexcept {
  return right == 0u || left <= std::numeric_limits<std::size_t>::max() / right;
}

[[nodiscard]] constexpr bool multiply(const std::size_t left,
                                      const std::size_t right,
                                      std::size_t &value) noexcept {
  if (!multiply(left, right)) {
    return false;
  }
  value = left * right;
  return true;
}

[[nodiscard]] constexpr bool multiply(const std::size_t first,
                                      const std::size_t second,
                                      const std::size_t third,
                                      std::size_t &value) noexcept {
  std::size_t pair = 0u;
  if (!multiply(first, second, pair) || !multiply(pair, third)) {
    return false;
  }
  value = pair * third;
  return true;
}

} // namespace rund::compute::detail::size
