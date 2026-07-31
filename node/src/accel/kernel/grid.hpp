#pragma once

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

struct Grid final {
  std::uint32_t x{};
  std::uint32_t y{};

  [[nodiscard]] constexpr bool valid() const noexcept {
    return x != 0u && y != 0u;
  }
};

[[nodiscard]] constexpr Grid PlanGrid(const std::uint64_t work_items,
                                      const std::uint64_t group_width,
                                      const std::uint64_t max_x,
                                      const std::uint64_t max_y) noexcept {
  if (work_items == 0u || group_width == 0u || max_x == 0u || max_y == 0u) {
    return {};
  }
  const std::uint64_t groups = 1u + (work_items - 1u) / group_width;
  const std::uint64_t x = groups < max_x ? groups : max_x;
  const std::uint64_t y = 1u + (groups - 1u) / x;
  constexpr std::uint64_t u32 = std::numeric_limits<std::uint32_t>::max();
  return x <= u32 && y <= max_y && y <= u32
             ? Grid{.x = static_cast<std::uint32_t>(x),
                    .y = static_cast<std::uint32_t>(y)}
             : Grid{};
}

} // namespace rund::node::accel::detail
