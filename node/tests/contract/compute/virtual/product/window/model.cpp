#include "local.hpp"

#include <algorithm>
#include <limits>

namespace rund_node_test_virtual::product::window_detail {

[[nodiscard]] bool
window_route_ok(const ProductRouteObservation &route) noexcept {
  return route.window_submit_accepted &&
         route.accepted_owner_mask == OwnerWindow &&
         route.accepted_owner_count == 1u && route.conflict_count == 0u;
}

[[nodiscard]] std::int32_t value(const std::size_t index) noexcept {
  return static_cast<std::int32_t>((index * 17u + 11u) % 41u);
}

[[nodiscard]] std::int32_t
expected_for(const std::size_t index,
             const std::size_t element_count) noexcept {
  std::int32_t total = 0;
  for (std::size_t sample = 0u; sample < Radius * 2u + 1u; ++sample) {
    const std::size_t raw = index + sample;
    const std::size_t selected = raw < Radius ? 0u
                                 : raw - Radius >= element_count
                                     ? element_count - 1u
                                     : raw - Radius;
    total += value(selected) + 3;
  }
  return total * 2;
}

[[nodiscard]] std::int32_t expected(const std::size_t index) noexcept {
  return expected_for(index, WindowElements);
}

[[nodiscard]] std::int32_t expected_clip_min(const std::size_t index) noexcept {
  std::int32_t minimum = std::numeric_limits<std::int32_t>::max();
  for (std::size_t sample = 0u; sample < Radius * 2u + 1u; ++sample) {
    const std::size_t raw = index + sample;
    if (raw >= Radius && raw - Radius < WindowElements) {
      minimum = std::min(minimum, value(raw - Radius));
    }
  }
  return minimum;
}

} // namespace rund_node_test_virtual::product::window_detail
