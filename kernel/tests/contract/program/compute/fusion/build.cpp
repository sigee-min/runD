#include "build/local.hpp"

namespace program_compute_contract {

int RunFusionBuildContract() {
  using namespace fusion_build_contract;
  if (RunCarrier() != 0 || RunPair() != 0 || RunChain() != 0 ||
      RunOutput() != 0 || RunShift() != 0) {
    return 1;
  }
  return RunCapacity();
}

} // namespace program_compute_contract
