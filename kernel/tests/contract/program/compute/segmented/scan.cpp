#include "scan/local.hpp"

namespace program_compute_contract {

int RunSegmentedScanContract() {
  if (SegmentedScanReject() != 0) {
    return 1;
  }
  if (SegmentedScanIdentity() != 0) {
    return 1;
  }
  if (SegmentedScanShape() != 0) {
    return 1;
  }
  if (SegmentedScanReference() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
