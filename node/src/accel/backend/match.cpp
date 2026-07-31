#include "match.hpp"

#include <string_view>

namespace rund::node::accel::detail {

bool SameReason(const char *const lhs, const char *const rhs) noexcept {
  return std::string_view{lhs == nullptr ? "" : lhs} ==
         std::string_view{rhs == nullptr ? "" : rhs};
}

bool SameCheck(const rund::AccelCheck &lhs,
               const rund::AccelCheck &rhs) noexcept {
  return lhs.ok == rhs.ok && SameReason(lhs.reason, rhs.reason) &&
         lhs.failed_batches == rhs.failed_batches &&
         lhs.first_failed_batch == rhs.first_failed_batch &&
         lhs.first_status == rhs.first_status;
}

bool SameCaps(const rund::kernel::ComputeCaps &lhs,
              const rund::kernel::ComputeCaps &rhs) noexcept {
  return lhs.api == rhs.api && lhs.device_bytes == rhs.device_bytes &&
         lhs.staging_bytes == rhs.staging_bytes &&
         lhs.max_window_tiles == rhs.max_window_tiles &&
         lhs.storage_alignment == rhs.storage_alignment &&
         lhs.subgroup_width == rhs.subgroup_width && lhs.ok == rhs.ok &&
         SameReason(lhs.reason, rhs.reason);
}

bool SameCpuCaps(const rund::kernel::CpuCaps &lhs,
                 const rund::kernel::CpuCaps &rhs) noexcept {
  return lhs.backend == rhs.backend && lhs.strategy == rhs.strategy &&
         lhs.lane_bytes == rhs.lane_bytes &&
         lhs.fixed_lane32_lanes == rhs.fixed_lane32_lanes &&
         lhs.fixed_lane64_lanes == rhs.fixed_lane64_lanes && lhs.ok == rhs.ok &&
         SameReason(lhs.reason, rhs.reason);
}

bool SameDispatch(const rund::kernel::ComputeBackendDispatch &lhs,
                  const rund::kernel::ComputeBackendDispatch &rhs) noexcept {
  return lhs.context == rhs.context && lhs.execute == rhs.execute &&
         lhs.last_error == rhs.last_error;
}

bool SameBackendInfo(const rund::AccelBackendInfo &lhs,
                     const rund::AccelBackendInfo &rhs) noexcept {
  return lhs.device_name == rhs.device_name &&
         lhs.driver_name == rhs.driver_name &&
         lhs.driver_info == rhs.driver_info &&
         lhs.storage_alignment == rhs.storage_alignment &&
         lhs.storage_bytes == rhs.storage_bytes;
}

} // namespace rund::node::accel::detail
