#include "contract/program/compute/fixed/arithmetic/local.hpp"

namespace program_compute_contract {

int RunComputeFixedArithmeticContract() {
  if (RunComputeFixedArithmeticDslContract() != 0) {
    return 1;
  }
  if (RunComputeFixedArithmeticLoweringContract() != 0) {
    return 1;
  }
  if (RunComputeFixedArithmeticRejectContract() != 0) {
    return 1;
  }
  return RunComputeFixedArithmeticFusionContract();
}

} // namespace program_compute_contract
