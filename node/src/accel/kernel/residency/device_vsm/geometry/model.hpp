#pragma once

#include <cstdint>

namespace rund::node::accel::detail {

inline constexpr char DeviceVsmDispatchCountOverflowReason[] =
    "device_vsm_dispatch_count_overflow";

struct DeviceVsmPageGeometry final {
  std::uint64_t logical_bytes{};
  std::uint64_t payload_bytes{};
  std::uint64_t frame_bytes{};
  std::uint64_t read_prefix_bytes{};
  std::uint64_t target_offset_bytes{};
  std::uint64_t read_suffix_bytes{};
  std::uint64_t page_count{};
  std::uint32_t element_bytes{};
};

struct DeviceVsmPageProjection final {
  std::uint64_t coordinate{};
  std::uint64_t core_offset{};
  std::uint64_t core_bytes{};
  std::uint64_t source_offset{};
  std::uint64_t source_bytes{};
  std::uint64_t frame_source_offset{};
};

struct DeviceVsmWindowFootprintAuthority final {
  std::uint64_t canonical_page_bytes{};
  std::uint64_t boundary_transition_count{};
  std::uint32_t projection_checksum{};

  [[nodiscard]] constexpr bool operator==(
      const DeviceVsmWindowFootprintAuthority &) const noexcept = default;
};

inline constexpr std::uint32_t DeviceVsmWindowRingSlotCount = 2u;
inline constexpr std::uint32_t DeviceVsmWindowRingConfigStride = 256u;

struct DeviceVsmWindowRingPlan final {
  bool gpu_owned{};
  std::uint32_t slot_count{};
  std::uint32_t page_count{};
  std::uint32_t frame_elements{};
  std::uint32_t payload_elements{};
  std::uint32_t halo_elements{};
  std::uint32_t schedule_checksum{};
  std::uint32_t config_stride{};
  std::uint64_t frame_bytes{};
  std::uint64_t state_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t config_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmWindowRingPlan &) const noexcept = default;
};

struct DeviceVsmWindowRingResult final {
  std::uint32_t page_count{};
  std::uint32_t schedule_checksum{};
  std::uint32_t round_trips{};

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmWindowRingResult &) const noexcept = default;
};

} // namespace rund::node::accel::detail
