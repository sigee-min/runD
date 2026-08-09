#include "window/local.hpp"

namespace program_compute_contract {

int RunWindowContract() {
  if (WindowReject() != 0) {
    return 1;
  }
  if (WindowIdentity() != 0) {
    return 1;
  }
  if (WindowShape() != 0) {
    return 1;
  }
  if (WindowReference() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
