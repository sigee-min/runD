#include "contract/program/compute/graph/local.hpp"

namespace program_compute_contract {

int RunGraphContract() {
  if (RunGraphIdentityContract() != 0) {
    return 1;
  }
  if (RunGraphAdmissionContract() != 0) {
    return 1;
  }
  if (RunGraphSignatureContract() != 0) {
    return 1;
  }
  return RunGraphRejectionContract();
}

}  // namespace program_compute_contract
