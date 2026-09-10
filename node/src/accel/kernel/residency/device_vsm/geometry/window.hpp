#pragma once

#include "ring.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline bool device_vsm_project_window_footprint(
    const DeviceVsmPageGeometry &geometry,
    DeviceVsmWindowFootprintAuthority &authority) noexcept {
  if (!device_vsm_runtime_geometry_valid(geometry) ||
      !device_vsm_centered_window_geometry(geometry)) {
    return false;
  }
  const std::uint64_t pages = geometry.page_count;
  const std::uint64_t logical_elements =
      geometry.logical_bytes / geometry.element_bytes;
  const std::uint64_t first_page_sum =
      pages <= 2u ? 0u : device_vsm_triangular(pages - 2u);
  const std::uint64_t last_page_sum =
      device_vsm_triangular(pages - 1u) + (pages - 1u);
  const std::uint64_t epoch_sum = device_vsm_triangular(pages);
  const auto multiply_modulo = [](const std::uint32_t factor,
                                  const std::uint64_t value) noexcept {
    return factor * static_cast<std::uint32_t>(value);
  };
  const std::uint32_t checksum = multiply_modulo(3u, epoch_sum) +
                                 multiply_modulo(5u, first_page_sum) +
                                 multiply_modulo(7u, last_page_sum) +
                                 multiply_modulo(11u, logical_elements);
  authority = DeviceVsmWindowFootprintAuthority{
      .canonical_page_bytes = geometry.payload_bytes,
      .boundary_transition_count = pages - 1u,
      .projection_checksum = checksum,
  };
  return true;
}

} // namespace rund::node::accel::detail
