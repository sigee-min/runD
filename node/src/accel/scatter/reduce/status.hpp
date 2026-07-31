#pragma once

#include <accel/check.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr rund::AccelCheck
ScatterReduceStatus(const std::uint32_t status) noexcept {
  if (status == 1u) {
    return {false, "compute_scatter_reduce_count_out_of_range"};
  }
  if (status == 2u) {
    return {false, "compute_scatter_reduce_index_out_of_range"};
  }
  return status == 0u
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_scatter_reduce_buffer_invalid"};
}

} // namespace rund::node::accel::detail
