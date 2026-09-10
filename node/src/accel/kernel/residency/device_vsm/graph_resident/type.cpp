#include "../graph_resident.hpp"

#include <rund/compute/fixed.hpp>

namespace rund::node::accel::detail {

bool device_vsm_graph_resident_type_valid(
    const DeviceVsmGraphResidentType &type) noexcept {
  return (type.scalar == rund::kernel::ComputeScalar::Lane32 &&
          type.domain == rund::kernel::ComputeDomain::U32 &&
          type.element_bytes == sizeof(std::uint32_t)) ||
         (type.scalar == rund::kernel::ComputeScalar::Lane64 &&
          type.domain == rund::kernel::ComputeDomain::U64 &&
          type.element_bytes == sizeof(std::uint64_t));
}

std::uint8_t device_vsm_graph_resident_type_code(
    const DeviceVsmGraphResidentType &type) noexcept {
  return type.domain == rund::kernel::ComputeDomain::U32
             ? static_cast<std::uint8_t>(rund::compute::detail::Type::U32)
             : static_cast<std::uint8_t>(rund::compute::detail::Type::U64);
}

} // namespace rund::node::accel::detail
