#include "scan/local.hpp"

namespace program_compute_contract {

int RunScanContract() {
  if (ScanReject() != 0) {
    return 1;
  }
  if (ScanIdentity() != 0) {
    return 1;
  }
  if (ScanShape() != 0) {
    return 1;
  }
  if (ScanReference() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
