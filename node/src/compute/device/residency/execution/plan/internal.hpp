#pragma once

#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency::execution::plan_internal {

[[nodiscard]] constexpr bool checked_add(const std::uint64_t left,
                                         const std::uint64_t right,
                                         std::uint64_t &result) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] constexpr bool checked_mul(const std::uint64_t left,
                                         const std::uint64_t right,
                                         std::uint64_t &result) noexcept {
  if (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

} // namespace rund::compute::detail::residency::execution::plan_internal
