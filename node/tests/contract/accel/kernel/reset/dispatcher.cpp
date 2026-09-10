#include "local.hpp"

namespace node_accel_contract {

bool ResetModelContract() {
  return reset_contract::CheckWidthAndLayout() &&
         reset_contract::CheckInvalidRanges() &&
         reset_contract::CheckVulkanExecution() &&
         reset_contract::CheckUniqueProjection() &&
         reset_contract::CheckLifetimeOverlap() &&
         reset_contract::CheckSealedPlans() &&
         reset_contract::CheckProjectionMatrix() &&
         reset_contract::CheckProjectionAmbiguity();
}

} // namespace node_accel_contract
