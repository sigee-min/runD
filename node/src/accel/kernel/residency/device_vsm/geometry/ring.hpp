#pragma once

#include "page.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] inline std::uint64_t
device_vsm_triangular(const std::uint64_t value) noexcept {
  return value % 2u == 0u ? (value / 2u) * (value + 1u)
                          : value * ((value + 1u) / 2u);
}

[[nodiscard]] inline bool
device_vsm_window_ring_plan_expected(const DeviceVsmPageGeometry &geometry,
                                     DeviceVsmWindowRingPlan &plan) noexcept {
  plan = {};
  if (!device_vsm_runtime_geometry_valid(geometry) ||
      !device_vsm_centered_window_geometry(geometry) ||
      geometry.element_bytes != sizeof(std::uint32_t) ||
      geometry.page_count < DeviceVsmWindowRingSlotCount ||
      geometry.page_count > std::numeric_limits<std::uint32_t>::max() / 2u) {
    return false;
  }
  const std::uint64_t frame_elements =
      geometry.frame_bytes / geometry.element_bytes;
  const std::uint64_t payload_elements =
      geometry.payload_bytes / geometry.element_bytes;
  const std::uint64_t halo_elements =
      geometry.read_prefix_bytes / geometry.element_bytes;
  if (frame_elements == 0u || payload_elements == 0u ||
      frame_elements > std::numeric_limits<std::uint32_t>::max() ||
      payload_elements > std::numeric_limits<std::uint32_t>::max() ||
      halo_elements > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  std::uint64_t state_bytes = 0u;
  std::uint64_t scratch_bytes = 0u;
  std::uint64_t config_bytes = 0u;
  if (!::rund::kernel::checked::mul(DeviceVsmWindowRingSlotCount,
                                    sizeof(std::uint32_t), state_bytes) ||
      !::rund::kernel::checked::mul(DeviceVsmWindowRingSlotCount,
                                    geometry.frame_bytes, scratch_bytes) ||
      !::rund::kernel::checked::mul(1u, DeviceVsmWindowRingConfigStride,
                                    config_bytes)) {
    return false;
  }
  const std::uint64_t q = geometry.page_count;
  const std::uint64_t half = q / DeviceVsmWindowRingSlotCount;
  const std::uint64_t slots =
      q % DeviceVsmWindowRingSlotCount == 0u ? half * (half - 1u) : half * half;
  const std::uint64_t checksum =
      3u * device_vsm_triangular(q) + 5u * slots +
      7u * (geometry.logical_bytes / geometry.element_bytes) +
      11u * (geometry.frame_bytes / geometry.element_bytes) * q;
  plan = DeviceVsmWindowRingPlan{
      .gpu_owned = true,
      .slot_count = DeviceVsmWindowRingSlotCount,
      .page_count = static_cast<std::uint32_t>(q),
      .frame_elements = static_cast<std::uint32_t>(frame_elements),
      .payload_elements = static_cast<std::uint32_t>(payload_elements),
      .halo_elements = static_cast<std::uint32_t>(halo_elements),
      .schedule_checksum = static_cast<std::uint32_t>(checksum),
      .config_stride = DeviceVsmWindowRingConfigStride,
      .frame_bytes = geometry.frame_bytes,
      .state_bytes = state_bytes,
      .scratch_bytes = scratch_bytes,
      .config_bytes = config_bytes,
  };
  return true;
}

[[nodiscard]] inline bool device_vsm_window_internal_dispatch_count(
    const std::uint64_t page_count, std::uint64_t &dispatch_count) noexcept {
  if (!::rund::kernel::checked::mul(2u, page_count, dispatch_count)) {
    dispatch_count = 0u;
    return false;
  }
  return true;
}

[[nodiscard]] inline bool device_vsm_window_internal_dispatch_count_native(
    const std::uint64_t page_count, std::uint32_t &dispatch_count) noexcept {
  std::uint64_t exact = 0u;
  if (!device_vsm_window_internal_dispatch_count(page_count, exact) ||
      exact > std::numeric_limits<std::uint32_t>::max()) {
    dispatch_count = 0u;
    return false;
  }
  dispatch_count = static_cast<std::uint32_t>(exact);
  return true;
}

[[nodiscard]] inline bool device_vsm_window_ring_result_expected(
    const DeviceVsmPageGeometry &geometry, const DeviceVsmWindowRingPlan &plan,
    DeviceVsmWindowRingResult &result) noexcept {
  result = {};
  DeviceVsmWindowRingPlan expected{};
  if (!device_vsm_window_ring_plan_expected(geometry, expected) ||
      expected != plan || plan.page_count == 0u ||
      plan.page_count != geometry.page_count) {
    return false;
  }
  result = DeviceVsmWindowRingResult{
      .page_count = plan.page_count,
      .schedule_checksum = plan.schedule_checksum,
      .round_trips = plan.page_count,
  };
  return true;
}

[[nodiscard]] inline bool device_vsm_ring_schedule_expected(
    const DeviceVsmPageGeometry &geometry, const std::uint32_t width,
    std::uint32_t &reuse_transitions, std::uint32_t &checksum) noexcept {
  if (!device_vsm_runtime_geometry_valid(geometry) || width < 2u ||
      width > 4u || width > geometry.page_count) {
    reuse_transitions = 0u;
    checksum = 0u;
    return false;
  }
  const std::uint64_t pages = geometry.page_count;
  const std::uint64_t full_turns = pages / width;
  const std::uint64_t remainder = pages % width;
  const std::uint64_t page_sum = device_vsm_triangular(pages);
  const std::uint64_t slot_sum = full_turns * device_vsm_triangular(width) +
                                 device_vsm_triangular(remainder);
  const std::uint64_t turn_sum =
      width * device_vsm_triangular(full_turns) + remainder * (full_turns + 1u);
  const std::uint64_t logical_elements =
      geometry.logical_bytes / geometry.element_bytes;
  const auto multiply_modulo = [](const std::uint32_t factor,
                                  const std::uint64_t value) noexcept {
    return factor * static_cast<std::uint32_t>(value);
  };
  reuse_transitions = static_cast<std::uint32_t>(pages - width);
  checksum = multiply_modulo(3u, page_sum) + multiply_modulo(5u, slot_sum) +
             multiply_modulo(7u, turn_sum) +
             multiply_modulo(11u, logical_elements);
  return true;
}

[[nodiscard]] inline bool device_vsm_ring_storage_expected(
    const DeviceVsmPageGeometry &geometry, const std::uint32_t width,
    const std::uint32_t region_count, std::uint64_t &state_bytes,
    std::uint64_t &scratch_bytes) noexcept {
  state_bytes = 0u;
  scratch_bytes = 0u;
  if (!device_vsm_runtime_geometry_valid(geometry) || width < 2u ||
      width > 4u || width > geometry.page_count || region_count < 2u ||
      (geometry.element_bytes != sizeof(std::uint32_t) &&
       geometry.element_bytes != sizeof(std::uint64_t))) {
    return false;
  }
  std::uint64_t region_bytes = 0u;
  return ::rund::kernel::checked::mul(width, sizeof(std::uint32_t),
                                      state_bytes) &&
         ::rund::kernel::checked::mul(width, geometry.payload_bytes,
                                      region_bytes) &&
         ::rund::kernel::checked::mul(region_count, region_bytes,
                                      scratch_bytes);
}

[[nodiscard]] inline bool
device_vsm_ring_round_trips_expected(const DeviceVsmPageGeometry &geometry,
                                     const std::uint32_t region_count,
                                     std::uint32_t &round_trips) noexcept {
  round_trips = 0u;
  std::uint64_t expected = 0u;
  if (!device_vsm_runtime_geometry_valid(geometry) || region_count < 2u ||
      !::rund::kernel::checked::mul(geometry.page_count, region_count,
                                    expected) ||
      expected > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  round_trips = static_cast<std::uint32_t>(expected);
  return true;
}

} // namespace rund::node::accel::detail
