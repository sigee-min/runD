#include "contract/program/compute/backend/lowering/reject/local.hpp"

namespace program_compute_contract {

int RunComputeBackendLoweringRejectContract() {
  if (lowering_reject::Shape() != 0) {
    return 1;
  }
  if (lowering_reject::Storage() != 0) {
    return 1;
  }
  if (lowering_reject::Binding() != 0) {
    return 1;
  }
  if (lowering_reject::Carrier() != 0) {
    return 1;
  }
  return lowering_reject::Domain();
}

} // namespace program_compute_contract
