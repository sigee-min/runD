#include "contract/program/skeleton/shape/local.hpp"

namespace program_skeleton_contract {

int RunSkeletonShapeContract() {
  if (SkeletonShapeStatic() != 0) {
    return 1;
  }
  if (SkeletonShapeElementwise() != 0) {
    return 1;
  }
  if (SkeletonShapeFold() != 0) {
    return 1;
  }
  if (SkeletonShapeViews() != 0) {
    return 1;
  }
  return SkeletonShapeCallbacks();
}

} // namespace program_skeleton_contract
