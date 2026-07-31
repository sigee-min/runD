#include "gather/local.hpp"

namespace program_compute_contract {

int RunGatherContract() {
  if (GatherReject() != 0) {
    return 1;
  }
  if (GatherIdentity() != 0) {
    return 1;
  }
  if (GatherShape() != 0) {
    return 1;
  }
  if (GatherReference() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
