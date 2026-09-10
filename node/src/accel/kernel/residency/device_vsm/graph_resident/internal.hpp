#pragma once

#include "../graph_resident.hpp"

#include <array>

namespace rund::node::accel::detail::device_vsm_graph_resident {

struct ValidationContext final {
  std::uint64_t payload_elements{};
  std::array<bool, DeviceVsmGraphResidentPhysicalCapacity> owner_used{};
  std::size_t port_total{};
};

[[nodiscard]] bool validate_root(
    const DeviceVsmGraphResidentProof &, std::uint64_t supplied_payload_elements,
    std::uint64_t page_count, ValidationContext &) noexcept;

[[nodiscard]] bool validate_owners(const DeviceVsmGraphResidentProof &,
                                   const ValidationContext &) noexcept;

[[nodiscard]] bool validate_owner_usage(const DeviceVsmGraphResidentProof &,
                                        const ValidationContext &) noexcept;

[[nodiscard]] bool validate_resources(
    const DeviceVsmGraphResidentProof &, std::uint32_t external_input_count,
    ValidationContext &) noexcept;

[[nodiscard]] bool validate_stages(const DeviceVsmGraphResidentProof &,
                                   ValidationContext &) noexcept;

[[nodiscard]] bool validate_lifetimes(const DeviceVsmGraphResidentProof &,
                                      const ValidationContext &) noexcept;

[[nodiscard]] bool validate_tails(const DeviceVsmGraphResidentProof &) noexcept;

} // namespace rund::node::accel::detail::device_vsm_graph_resident
