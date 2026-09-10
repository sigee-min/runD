#include "local.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::measure::compute::route_matrix::oracle {

namespace oracle_detail {

[[nodiscard]] std::int32_t window_value(const std::uint64_t index,
                                        const std::uint64_t count,
                                        const std::uint64_t radius) noexcept {
  std::int32_t result = 0;
  for (std::uint64_t delta = 0u; delta <= radius * 2u; ++delta) {
    const std::uint64_t raw = index + delta;
    const std::uint64_t selected =
        raw < radius ? 0u : std::min(raw - radius, count - 1u);
    result += seed_value(selected);
  }
  return result;
}

[[nodiscard]] bool mul(const std::uint64_t left, const std::uint64_t right,
                       std::uint64_t &value) noexcept {
  if (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  value = left * right;
  return true;
}

} // namespace oracle_detail

} // namespace rund::measure::compute::route_matrix::oracle
