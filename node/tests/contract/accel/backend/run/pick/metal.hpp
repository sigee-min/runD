#pragma once

#include "fake.hpp"

namespace node_accel_contract::backend_pick {

[[nodiscard]] inline bool MetalIdentityContract() {
  const rund::AccelDevice pick = Pick({rund::AccelApi::Metal});
  if (!pick.check.ok) {
    return MetalFailsClosed(pick);
  }
  return pick.api == rund::AccelApi::Metal &&
         !pick.backend_info.device_name.empty() &&
         pick.backend_info.driver_name == "Metal" &&
         pick.backend_info.driver_info.empty() &&
         rund::kernel::ComputeStorageAlignmentValid(
             pick.caps.storage_alignment);
}

} // namespace node_accel_contract::backend_pick
