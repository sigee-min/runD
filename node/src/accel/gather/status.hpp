#pragma once

#include <accel/check.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr rund::AccelCheck
GatherStatus(const std::uint32_t status) noexcept {
  if (status == 1u) {
    return {false, "compute_bounded_count_invalid"};
  }
  if (status == 2u) {
    return {false, "compute_gather_index_out_of_range"};
  }
  return status == 0u ? rund::AccelCheck{true, "ok"}
                      : rund::AccelCheck{false, "compute_gather_invalid"};
}

} // namespace rund::node::accel::detail
