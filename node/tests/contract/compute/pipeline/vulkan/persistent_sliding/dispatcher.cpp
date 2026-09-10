#include "local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU) ||                                     \
    defined(RUND_NODE_TEST_BACKEND_METAL) ||                                   \
    !defined(RUND_NODE_HAVE_VULKAN_SDK)

int RunComputePipelineVulkanPersistentSlidingContract() { return 0; }

#else

namespace persistent = rund_node_test_pipeline_vulkan_persistent;

int RunComputePipelineVulkanPersistentSlidingContract() {
  if (!rund_node_test_pipeline_residency::device_vsm_test::CheckActualDeviceVsm(
          rund::compute::Backend::Vulkan,
          rund::node::accel::detail::PrepareVulkanDeviceVsm,
          persistent::DeviceVsmQueueCount)) {
    std::fprintf(stderr, "Vulkan DeviceVsm actual contract failed\n");
    return 1;
  }
  const int device_vsm_product =
      rund_node_test_device_vsm_product::CheckDeviceVsmProduct(
          rund::compute::Backend::Vulkan, persistent::ProductQueueCount);
  if (device_vsm_product != 0) {
    return device_vsm_product;
  }
  if (!persistent::CheckFusedDirectHistory()) {
    return 1;
  }
  if (!persistent::CheckFusedDirectRecurrence()) {
    return 1;
  }
  if (!persistent::CheckGeneratedReadAtMap()) {
    return 1;
  }
  std::uint64_t identity = 1u;
  for (const std::uint64_t coordinate_count : {5u, 9u, 257u}) {
    if (!persistent::RunPersistentCase(coordinate_count, identity++, false,
                                       false, false, false)) {
      std::fprintf(stderr, "Vulkan persistent sliding Q=%llu failed\n",
                   static_cast<unsigned long long>(coordinate_count));
      return 1;
    }
  }
  if (!persistent::RunPersistentCase(5u, identity++, true, false, false,
                                     false)) {
    std::fprintf(stderr, "Vulkan persistent sliding Known failure failed\n");
    return 1;
  }
  if (!persistent::RunPersistentCase(5u, identity++, false, false, true,
                                     false)) {
    std::fprintf(stderr,
                 "Vulkan persistent sliding failed-admission-only failed\n");
    return 1;
  }
  if (!persistent::RunPersistentCase(5u, identity++, false, false, false,
                                     true)) {
    std::fprintf(stderr,
                 "Vulkan persistent sliding submit device-loss failed\n");
    return 1;
  }
  const int product = rund_node_test_persistent_product::CheckPersistentProduct(
      rund::compute::Backend::Vulkan, persistent::ProductQueueCount);
  if (product != 0) {
    return product;
  }
  if (!persistent::RunPersistentCase(5u, identity++, false, true, false,
                                     false)) {
    std::fprintf(stderr, "Vulkan persistent sliding Unknown failure failed\n");
    return 1;
  }
  return 0;
}

#endif
