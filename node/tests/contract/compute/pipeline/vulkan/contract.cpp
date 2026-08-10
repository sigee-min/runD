#include "../local.hpp"

#include <cstdio>

int RunComputePipelineVulkanTransferContract() {
  const int result = rund_node_test_pipeline::CheckVulkanTransferAdmission();
  if (result != 0) {
    std::fprintf(stderr, "pipeline Vulkan transfer admission result=%d\n",
                 result);
  }
  return result;
}
