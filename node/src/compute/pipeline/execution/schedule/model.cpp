#include "local.hpp"

#include <limits>

namespace rund::compute::detail {

bool pipeline_schedule_generation_for(
    const node::accel::detail::PreparedResidencyScheduleRole &role,
    const std::uint64_t epoch, std::uint32_t &generation) noexcept {
  const std::uint64_t turn =
      epoch / node::accel::detail::ResidencyScheduleRoleCapacity;
  const std::uint64_t stride = role.control_generation_stride;
  if (turn != 0u && stride > std::numeric_limits<std::uint64_t>::max() / turn) {
    return false;
  }
  const std::uint64_t delta = turn * stride;
  if (delta > std::numeric_limits<std::uint32_t>::max() ||
      role.first_control_generation >
          std::numeric_limits<std::uint32_t>::max() - delta) {
    return false;
  }
  generation =
      role.first_control_generation + static_cast<std::uint32_t>(delta);
  return true;
}

} // namespace rund::compute::detail
