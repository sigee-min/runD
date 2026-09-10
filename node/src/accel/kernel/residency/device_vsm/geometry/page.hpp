#pragma once

#include "model.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool device_vsm_complete_frame_geometry(
    const DeviceVsmPageGeometry &geometry) noexcept {
  return geometry.read_prefix_bytes == 0u &&
         geometry.target_offset_bytes == 0u &&
         geometry.read_suffix_bytes == 0u &&
         geometry.frame_bytes == geometry.payload_bytes;
}

[[nodiscard]] inline bool device_vsm_centered_window_geometry(
    const DeviceVsmPageGeometry &geometry) noexcept {
  std::uint64_t complete = 0u;
  return geometry.read_prefix_bytes != 0u &&
         geometry.read_prefix_bytes == geometry.target_offset_bytes &&
         geometry.read_prefix_bytes == geometry.read_suffix_bytes &&
         ::rund::kernel::checked::add(geometry.target_offset_bytes,
                                      geometry.payload_bytes, complete) &&
         ::rund::kernel::checked::add(complete, geometry.read_suffix_bytes,
                                      complete) &&
         complete == geometry.frame_bytes;
}

[[nodiscard]] inline bool
device_vsm_geometry_valid(const DeviceVsmPageGeometry &geometry) noexcept {
  if (geometry.logical_bytes == 0u || geometry.payload_bytes == 0u ||
      geometry.frame_bytes == 0u || geometry.page_count < 2u ||
      geometry.element_bytes == 0u ||
      geometry.logical_bytes % geometry.element_bytes != 0u ||
      geometry.payload_bytes % geometry.element_bytes != 0u ||
      geometry.frame_bytes % geometry.element_bytes != 0u ||
      geometry.read_prefix_bytes % geometry.element_bytes != 0u ||
      geometry.target_offset_bytes % geometry.element_bytes != 0u ||
      geometry.read_suffix_bytes % geometry.element_bytes != 0u ||
      geometry.read_prefix_bytes > geometry.target_offset_bytes ||
      geometry.read_prefix_bytes > geometry.payload_bytes ||
      geometry.read_suffix_bytes > geometry.payload_bytes) {
    return false;
  }
  std::uint64_t payload_span = 0u;
  std::uint64_t complete_span = 0u;
  if (!::rund::kernel::checked::add(geometry.target_offset_bytes,
                                    geometry.payload_bytes, payload_span) ||
      !::rund::kernel::checked::add(payload_span, geometry.read_suffix_bytes,
                                    complete_span) ||
      complete_span > geometry.frame_bytes) {
    return false;
  }
  const std::uint64_t quotient =
      geometry.logical_bytes / geometry.payload_bytes;
  const std::uint64_t remainder =
      geometry.logical_bytes % geometry.payload_bytes;
  const std::uint64_t expected_pages = quotient + (remainder != 0u ? 1u : 0u);
  return expected_pages == geometry.page_count;
}

[[nodiscard]] inline bool device_vsm_runtime_geometry_valid(
    const DeviceVsmPageGeometry &geometry) noexcept {
  if (!device_vsm_geometry_valid(geometry)) {
    return false;
  }
  constexpr std::uint64_t Maximum = std::numeric_limits<std::uint32_t>::max();
  return geometry.page_count <= Maximum &&
         geometry.logical_bytes / geometry.element_bytes <= Maximum &&
         geometry.payload_bytes / geometry.element_bytes <= Maximum &&
         geometry.frame_bytes / geometry.element_bytes <= Maximum;
}

[[nodiscard]] inline bool
device_vsm_project_page(const DeviceVsmPageGeometry &geometry,
                        const std::uint64_t coordinate,
                        DeviceVsmPageProjection &projection) noexcept {
  if (!device_vsm_geometry_valid(geometry) ||
      coordinate >= geometry.page_count) {
    return false;
  }
  std::uint64_t core_offset = 0u;
  if (!::rund::kernel::checked::mul(coordinate, geometry.payload_bytes,
                                    core_offset) ||
      core_offset >= geometry.logical_bytes) {
    return false;
  }
  const std::uint64_t core_bytes =
      std::min(geometry.payload_bytes, geometry.logical_bytes - core_offset);
  const std::uint64_t prefix =
      std::min(geometry.read_prefix_bytes, core_offset);
  const std::uint64_t core_end = core_offset + core_bytes;
  const std::uint64_t suffix =
      std::min(geometry.read_suffix_bytes, geometry.logical_bytes - core_end);
  std::uint64_t source_bytes = 0u;
  if (!::rund::kernel::checked::add(prefix, core_bytes, source_bytes) ||
      !::rund::kernel::checked::add(source_bytes, suffix, source_bytes)) {
    return false;
  }
  projection = DeviceVsmPageProjection{
      .coordinate = coordinate,
      .core_offset = core_offset,
      .core_bytes = core_bytes,
      .source_offset = core_offset - prefix,
      .source_bytes = source_bytes,
      .frame_source_offset = geometry.target_offset_bytes - prefix,
  };
  return projection.frame_source_offset <= geometry.frame_bytes &&
         projection.source_bytes <=
             geometry.frame_bytes - projection.frame_source_offset;
}

[[nodiscard]] inline bool
device_vsm_overlap_reuse_bytes(const DeviceVsmPageGeometry &geometry,
                               std::uint64_t &bytes) noexcept {
  if (!device_vsm_geometry_valid(geometry)) {
    return false;
  }
  std::uint64_t per_transition = 0u;
  return ::rund::kernel::checked::add(geometry.read_prefix_bytes,
                                      geometry.read_suffix_bytes,
                                      per_transition) &&
         ::rund::kernel::checked::mul(geometry.page_count - 1u, per_transition,
                                      bytes);
}

} // namespace rund::node::accel::detail
