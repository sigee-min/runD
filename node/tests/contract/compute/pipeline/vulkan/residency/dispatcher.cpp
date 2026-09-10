#include "../../local.hpp"
#include "internal.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU) || \
    defined(RUND_NODE_TEST_BACKEND_METAL) || \
    !defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline {

int CheckVulkanResidencyWindow() { return 0; }

} // namespace rund_node_test_pipeline

#else

namespace rund_node_test_pipeline {

int CheckVulkanResidencyWindow() {
  rund_node_test_pipeline_vulkan_residency::ResidencyFixture fixture{};
  const int initialized =
      rund_node_test_pipeline_vulkan_residency::InitializeResidencyFixture(
          fixture);
  if (initialized != 0 || fixture.skipped) {
    return initialized;
  }
  using namespace rund_node_test_pipeline_vulkan_residency;
  if (const int window = CheckWindow(fixture); window != 0) {
    return window;
  }
  if (const int schedule = CheckSchedule(fixture); schedule != 0) {
    return schedule;
  }
  if (const int warm = CheckWarm(fixture); warm != 0) {
    return warm;
  }
  return CheckAbort(fixture);
}

} // namespace rund_node_test_pipeline

#endif
