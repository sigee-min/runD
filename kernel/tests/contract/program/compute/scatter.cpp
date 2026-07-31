#include "scatter/local.hpp"

namespace program_compute_contract {

int RunScatterContract() {
  if (ScatterReject() != 0) {
    return 1;
  }
  if (ScatterIdentity() != 0) {
    return 1;
  }
  if (ScatterShape() != 0) {
    return 1;
  }
  if (ScatterReference() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
