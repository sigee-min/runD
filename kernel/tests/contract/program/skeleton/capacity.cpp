#include "capacity/local.hpp"

namespace program_skeleton_contract {

int RunSkeletonCapacityContract() {
  if (test_skeleton_scheduled_executor_capacity() != 0) {
    return 1;
  }
  if (test_skeleton_invalid_executor_and_policy_rejections() != 0) {
    return 1;
  }
  if (test_skeleton_policy_runtime() != 0) {
    return 1;
  }
  if (test_skeleton_explicit_physical_tile_policy_report() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_skeleton_contract
