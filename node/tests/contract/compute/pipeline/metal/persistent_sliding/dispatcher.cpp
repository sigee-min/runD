#include "local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU) || \
    defined(RUND_NODE_TEST_BACKEND_VULKAN)

int RunComputePipelineMetalPersistentSlidingContract() { return 0; }

#elif !defined(RUND_NODE_HAVE_METAL_SDK)

int RunComputePipelineMetalPersistentSlidingContract() {
  const auto opened = rund::compute::open(rund::compute::Target::metal());
  if (opened || opened.reason() != rund::compute::Reason::AdapterUnavailable) {
    std::fprintf(stderr, "Metal SDK-disabled open contract failed\n");
    return 1;
  }
  std::fprintf(stderr,
               "Metal SDK unavailable: open rejected; native execution not run\n");
  return 0;
}

#else

namespace persistent = rund_node_test_pipeline_metal_persistent;

int RunComputePipelineMetalPersistentSlidingContract() {
  if (!rund_node_test_pipeline_residency::device_vsm_test::CheckActualDeviceVsm(
          rund::compute::Backend::Metal,
          rund::node::accel::detail::PrepareMetalDeviceVsm,
          persistent::DeviceVsmQueueCount)) {
    std::fprintf(stderr, "Metal DeviceVsm actual contract failed\n");
    return 1;
  }
  const int device_vsm_product =
      rund_node_test_device_vsm_product::CheckDeviceVsmProduct(
          rund::compute::Backend::Metal, persistent::ProductQueueCount);
  if (device_vsm_product != 0) {
    return device_vsm_product;
  }
  const int persistent_product =
      rund_node_test_persistent_product::CheckPersistentProduct(
          rund::compute::Backend::Metal, persistent::ProductQueueCount);
  if (persistent_product != 0) {
    return persistent_product;
  }
  if (!persistent::CheckFusedDirectHistory()) {
    return 1;
  }
  if (!persistent::CheckFusedDirectRecurrence()) {
    return 1;
  }
  std::uint64_t identity = 1u;
  for (const std::uint64_t coordinate_count : {5u, 9u, 257u}) {
    if (!persistent::RunPersistentCase(coordinate_count, identity++, false,
                                        false, false)) {
      std::fprintf(stderr, "Metal persistent sliding Q=%llu failed\n",
                   static_cast<unsigned long long>(coordinate_count));
      return 1;
    }
    std::fprintf(stderr,
                 "legacy persistent Q=%llu native_submit=1 "
                 "epoch_submit=0 epoch_callback=0 final=1\n",
                 static_cast<unsigned long long>(coordinate_count));
  }
  if (!persistent::RunPersistentCase(5u, identity++, true, false, false)) {
    std::fprintf(stderr, "Metal persistent sliding Known failure failed\n");
    return 1;
  }
  if (!persistent::RunPersistentCase(5u, identity++, false, false, true)) {
    std::fprintf(stderr,
                 "Metal persistent sliding failed-admission-only failed\n");
    return 1;
  }
  if (!persistent::RunPublicSpatialWindowN48()) {
    std::fprintf(stderr, "Metal public spatial Window product failed\n");
    return 1;
  }
  if (!persistent::RunNaturalFallbackWindowN53()) {
    std::fprintf(stderr, "Metal natural fallback Window N53 failed\n");
    return 1;
  }
  if (!persistent::RunPersistentCase(5u, identity++, false, true, false)) {
    std::fprintf(stderr, "Metal persistent sliding Unknown failure failed\n");
    return 1;
  }
  std::fprintf(stderr, "legacy persistent Known/failed-admission/Unknown "
                       "regressions passed\n");
  return 0;
}

#endif
