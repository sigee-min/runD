#include "skeleton/local.hpp"

int RunProgramSkeletonContract() {
  if (program_skeleton_contract::RunSkeletonShapeContract() != 0) {
    return 1;
  }
  if (program_skeleton_contract::RunSkeletonValidationContract() != 0) {
    return 1;
  }
  if (program_skeleton_contract::RunSkeletonCapacityContract() != 0) {
    return 1;
  }
  return 0;
}
