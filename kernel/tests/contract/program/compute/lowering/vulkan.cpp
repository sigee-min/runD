#include "contract/program/compute/lowering/vulkan/local.hpp"

namespace program_compute_contract {

int RunComputeVulkanLoweringContract() {
  if (VulkanLoweringBase() != 0) {
    return 1;
  }
  if (VulkanLoweringFixedLane64() != 0) {
    return 1;
  }
  if (VulkanLoweringExpanded() != 0) {
    return 1;
  }
  if (VulkanLoweringScalarOps() != 0) {
    return 1;
  }
  if (VulkanLoweringBitOps() != 0) {
    return 1;
  }
  if (VulkanLoweringNonlinearOps() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
