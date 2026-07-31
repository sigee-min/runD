#include "reduce/local.hpp"

namespace program_compute_contract {

int RunSegmentedReduceContract() {
  if (SegmentedReduceReject() != 0) {
    return 1;
  }
  if (SegmentedReduceIdentity() != 0) {
    return 1;
  }
  if (SegmentedReduceShape() != 0) {
    return 1;
  }
  if (SegmentedReduceReference() != 0) {
    return 1;
  }
  return 0;
}

}  // namespace program_compute_contract
