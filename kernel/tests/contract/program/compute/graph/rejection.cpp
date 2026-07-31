#include "contract/program/compute/graph/rejection/local.hpp"

namespace program_compute_contract {

int RunGraphRejectionContract() {
  if (GraphRejectPrimitive() != 0) {
    return 1;
  }
  if (GraphRejectNode() != 0) {
    return 1;
  }
  if (GraphRejectBuffer() != 0) {
    return 1;
  }
  return GraphRejectNumericPolicy();
}

} // namespace program_compute_contract
