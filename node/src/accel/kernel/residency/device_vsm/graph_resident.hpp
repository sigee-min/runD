#pragma once

#include "page_map.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/model.hpp>
#include <limits>
#include <memory>

namespace rund::node::accel::detail {

inline constexpr std::size_t DeviceVsmGraphResidentStageCapacity = 8u;
inline constexpr std::size_t DeviceVsmGraphResidentResourceCapacity = 9u;
inline constexpr std::size_t DeviceVsmGraphResidentPortCapacity = 16u;
inline constexpr std::size_t DeviceVsmGraphResidentPhysicalCapacity = 9u;
inline constexpr std::size_t DeviceVsmGraphResidentBankCapacity = 2u;
inline constexpr std::size_t DeviceVsmGraphResidentWidth = 2u;
inline constexpr std::uint32_t DeviceVsmGraphResidentExternalSlotInvalid =
    std::numeric_limits<std::uint32_t>::max();

// The planner's first resource type is sealed here once by
// project_graph_resident(). Resource rows, owners, refs, and generated table
// fields are consistency witnesses for this root and never select storage
// width independently.
struct DeviceVsmGraphResidentType final {
  rund::kernel::ComputeScalar scalar{};
  rund::kernel::ComputeDomain domain{};
  std::uint32_t element_bytes{};

  [[nodiscard]] constexpr bool operator==(
      const DeviceVsmGraphResidentType &) const noexcept = default;
};

struct DeviceVsmGraphResidentResource final {
  std::uint32_t resource{};
  std::uint32_t physical_id{};
  std::uint32_t color{};
  std::uint32_t producer{};
  std::uint32_t first{};
  std::uint32_t last{};
  std::uint32_t owner_slot{};
  std::uint32_t external_slot{};
  std::uint64_t page_bytes{};
  std::uint64_t logical_bytes{};
  std::uint8_t role{};
  std::uint8_t type{};
  rund::kernel::ComputeFixedFormat format{};
  std::uint8_t valid{};
};

struct DeviceVsmGraphResidentPort final {
  std::uint32_t resource{};
  std::uint32_t next_stage{};
  std::uint16_t program_port{};
  std::uint8_t access{};
  std::uint8_t valid{};
};

struct DeviceVsmGraphResidentStage final {
  std::array<DeviceVsmGraphResidentPort, DeviceVsmGraphResidentPortCapacity>
      ports{};
  std::uint32_t node{};
  std::uint16_t port_count{};
  std::uint8_t domain{};
  std::uint8_t valid{};
};

struct DeviceVsmGraphResidentRegion final {
  std::uint8_t tier{};
  std::uint8_t role{};
  std::uint32_t first{};
  std::uint32_t count{};

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmGraphResidentRegion &) const noexcept = default;
};

struct DeviceVsmGraphResidentOwner final {
  std::uint32_t physical_id{};
  std::uint32_t color{};
  std::uint64_t view{};
  std::array<std::uint32_t, DeviceVsmGraphResidentBankCapacity> local_first{};
  std::array<rund::kernel::ResidentBufferRef,
             DeviceVsmGraphResidentBankCapacity>
      refs{};
  std::array<std::shared_ptr<void>, DeviceVsmGraphResidentBankCapacity>
      handles{};
  std::array<DeviceVsmGraphResidentRegion, DeviceVsmGraphResidentBankCapacity>
      cache_regions{};
  std::array<DeviceVsmGraphResidentRegion, DeviceVsmGraphResidentBankCapacity>
      bank_regions{};
  std::array<std::uint64_t, DeviceVsmGraphResidentBankCapacity>
      buffer_generations{};
  rund::kernel::ComputeFixedFormat physical_format{};
  std::uint64_t physical_page_bytes{};
  std::uint8_t physical_type{};
  std::uint8_t physical_tier{};
  std::uint8_t physical_role{};
  std::uint8_t bank_count{};
  std::uint8_t valid{};
};

struct DeviceVsmGraphResidentProof final {
  std::array<DeviceVsmGraphResidentResource,
             DeviceVsmGraphResidentResourceCapacity>
      resources{};
  std::array<DeviceVsmGraphResidentStage, DeviceVsmGraphResidentStageCapacity>
      stages{};
  std::array<DeviceVsmGraphResidentOwner,
             DeviceVsmGraphResidentPhysicalCapacity>
      owners{};
  DeviceVsmPageMap page_map{};
  DeviceVsmGraphResidentType type{};
  std::uint64_t digest{};
  std::uint32_t resource_count{};
  std::uint32_t stage_count{};
  std::uint32_t port_count{};
  std::uint32_t owner_binding_count{};
  std::uint8_t width{DeviceVsmGraphResidentWidth};
  std::uint8_t valid{};
};

[[nodiscard]] bool device_vsm_graph_resident_type_valid(
    const DeviceVsmGraphResidentType &) noexcept;

[[nodiscard]] std::uint8_t device_vsm_graph_resident_type_code(
    const DeviceVsmGraphResidentType &) noexcept;

[[nodiscard]] bool device_vsm_graph_resident_same_object(
    const std::shared_ptr<void> &, const std::shared_ptr<void> &) noexcept;

[[nodiscard]] std::uint64_t device_vsm_graph_resident_digest(
    const DeviceVsmGraphResidentProof &) noexcept;

[[nodiscard]] bool device_vsm_graph_resident_valid(
    const DeviceVsmGraphResidentProof &,
    std::uint64_t supplied_payload_elements = 0u,
    std::uint32_t external_input_count =
        std::numeric_limits<std::uint32_t>::max(),
    std::uint64_t page_count = std::numeric_limits<std::uint64_t>::max()) noexcept;

} // namespace rund::node::accel::detail
