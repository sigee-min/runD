#include "local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU) || defined(RUND_NODE_TEST_BACKEND_METAL) || \
    !defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline {

int CheckVulkanTransferAdmission() { return 0; }

} // namespace rund_node_test_pipeline

#else

namespace rund_node_test_pipeline {

int CheckVulkanTransferAdmission() {
  using namespace vulkan_transfer;
  const int selected = ExactVulkanResidencySelection();
  if (selected != 0) {
    return 1000 + selected;
  }
  const int execution = ExactVulkanExecutionAdapter();
  if (execution != 0) {
    return 1100 + execution;
  }
  VulkanTransferProbe probe{};
  const int exact = ExactVulkanTransferProbe(probe);
  if (exact != 0) {
    return exact;
  }
  const int rollback = VulkanTransferBudgetRollback(probe);
  return rollback == 0 ? 0 : 100 + rollback;
}

} // namespace rund_node_test_pipeline

#endif
