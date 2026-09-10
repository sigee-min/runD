#include "reachability/local.hpp"

namespace program_compute_contract {

int RunComputeBackendHelperEmissionContract() {
  if (helper_emission::CheckIntegerHelpers() != 0) {
    return 1;
  }
  if (helper_emission::CheckFixedHelperSelection() != 0) {
    return 1;
  }
  if (helper_emission::CheckGenericU128Division() != 0) {
    return 1;
  }
  return helper_emission::CheckFixedHelperSourceSize();
}

} // namespace program_compute_contract
