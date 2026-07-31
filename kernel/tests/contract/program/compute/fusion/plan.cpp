#include "contract/program/compute/fusion/plan/local.hpp"

namespace program_compute_contract {

int RunFusionPlanContract() {
  if (RunFusionPlanSuccessContract() != 0) {
    return 1;
  }
  if (RunFusionPlanVisibilityContract() != 0) {
    return 1;
  }
  return RunFusionPlanRejectContract();
}

} // namespace program_compute_contract
