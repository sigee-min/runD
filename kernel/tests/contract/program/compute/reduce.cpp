#include "reduce/local.hpp"

namespace program_compute_contract {

int RunReduceContract() {
  if (ReduceReject() != 0) {
    return 1;
  }
  if (ReduceIdentity() != 0) {
    return 1;
  }
  if (ReduceShape() != 0) {
    return 1;
  }
  if (ReduceShapeOps() != 0) {
    return 1;
  }
  if (ReduceReference() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
