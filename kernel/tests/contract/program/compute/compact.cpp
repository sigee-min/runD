#include "compact/local.hpp"

namespace program_compute_contract {

int RunCompactContract() {
  if (CompactReject() != 0) {
    return 1;
  }
  if (CompactIdentity() != 0) {
    return 1;
  }
  if (CompactShape() != 0) {
    return 1;
  }
  if (CompactReference() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
