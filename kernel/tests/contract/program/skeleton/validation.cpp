#include "contract/program/skeleton/validation/local.hpp"

namespace program_skeleton_contract {

int RunSkeletonValidationContract() {
  if (SkeletonValidationView() != 0) {
    return 1;
  }
  if (SkeletonValidationPartition() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_skeleton_contract
