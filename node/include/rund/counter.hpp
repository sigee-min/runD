#pragma once

#include <cstdint>
#include <limits>

namespace rund::detail::counter {

[[nodiscard]] constexpr std::uint64_t
SaturatingAdd(const std::uint64_t left, const std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

constexpr void Accumulate(std::uint64_t &total,
                          const std::uint64_t value) noexcept {
  total = SaturatingAdd(total, value);
}

[[nodiscard]] constexpr std::uint64_t
Remaining(const std::uint64_t current,
          const std::uint64_t released) noexcept {
  return current == std::numeric_limits<std::uint64_t>::max()
             ? current
             : released > current ? 0u : current - released;
}

constexpr void Release(std::uint64_t &current,
                       const std::uint64_t released) noexcept {
  current = Remaining(current, released);
}

[[nodiscard]] constexpr std::uint64_t
Delta(const std::uint64_t before, const std::uint64_t after) noexcept {
  return before == std::numeric_limits<std::uint64_t>::max() ||
                 after == std::numeric_limits<std::uint64_t>::max()
             ? std::numeric_limits<std::uint64_t>::max()
             : after >= before ? after - before : 0u;
}

[[nodiscard]] constexpr std::uint64_t
SaturatingMultiply(const std::uint64_t left,
                   const std::uint64_t right) noexcept {
  return right != 0u && left > std::numeric_limits<std::uint64_t>::max() / right
             ? std::numeric_limits<std::uint64_t>::max()
             : left * right;
}

} // namespace rund::detail::counter
