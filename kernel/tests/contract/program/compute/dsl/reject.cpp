#include "contract/program/compute/dsl/reject/local.hpp"

namespace program_compute_contract {

int RunComputeDslRejectContract() {
  if (dsl_reject::Base() != 0) {
    return 1;
  }
  if (dsl_reject::Binding() != 0) {
    return 1;
  }
  if (dsl_reject::Storage() != 0) {
    return 1;
  }
  return dsl_reject::Domain();
}

} // namespace program_compute_contract
