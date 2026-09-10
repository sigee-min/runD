#include "local.hpp"

#include <cstdio>

int RunComputePipelineResidencyContract() {
  using namespace rund_node_test_pipeline_residency;
  if (const int result = rund_node_test_pipeline::CheckVulkanResidencyWindow();
      result != 0) {
    std::fprintf(stderr, "pipeline Vulkan residency window result=%d\n",
                 result);
    return 75 + result;
  }
  if (const int result = CheckPlanner(); result != 0) {
    std::fprintf(stderr, "pipeline residency planner result=%d\n", result);
    return 100 + result;
  }
  if (const int result = CheckIdentity(); result != 0) {
    std::fprintf(stderr, "pipeline residency identity result=%d\n", result);
    return 200 + result;
  }
  if (const int result = CheckGraphPlanner(); result != 0) {
    std::fprintf(stderr, "pipeline residency graph result=%d\n", result);
    return 300 + result;
  }
  if (const int result = CheckGraphProjection(); result != 0) {
    std::fprintf(stderr, "pipeline residency projection result=%d\n", result);
    return 340 + result;
  }
  if (const int result = CheckGraphForecast(); result != 0) {
    std::fprintf(stderr, "pipeline residency graph forecast result=%d\n",
                 result);
    return 350 + result;
  }
  if (const int result = CheckGraphPromote(); result != 0) {
    std::fprintf(stderr, "pipeline residency graph promote result=%d\n",
                 result);
    return 355 + result;
  }
  if (const int result = CheckGraphHostInputLayout(); result != 0) {
    std::fprintf(stderr, "pipeline residency graph Host input result=%d\n",
                 result);
    return 357 + result;
  }
  if (const int result = CheckGraphDrain(); result != 0) {
    std::fprintf(stderr, "pipeline residency graph drain result=%d\n", result);
    return 360 + result;
  }
  if (const int result = CheckGraphPersist(); result != 0) {
    std::fprintf(stderr, "pipeline residency graph persist result=%d\n",
                 result);
    return 365 + result;
  }
  if (const int result = CheckGraphWavefront(); result != 0) {
    std::fprintf(stderr, "pipeline residency graph wavefront result=%d\n",
                 result);
    return 375 + result;
  }
  if (const int result = CheckHostRingCapacities(); result != 0) {
    std::fprintf(stderr, "pipeline residency Host ring result=%d\n", result);
    return 390 + result;
  }
  if (const int result = CheckFootprints(); result != 0) {
    std::fprintf(stderr, "pipeline residency footprint result=%d\n", result);
    return 400 + result;
  }
  if (const int result = CheckAuthorityCache(); result != 0) {
    std::fprintf(stderr, "pipeline residency authority cache result=%d\n",
                 result);
    return 500 + result;
  }
  if (const int result = CheckAuthorityMigrationGraph(); result != 0) {
    std::fprintf(stderr,
                 "pipeline residency authority migration/graph result=%d\n",
                 result);
    return 550 + result;
  }
  if (const int result = CheckAuthorityRelocation(); result != 0) {
    std::fprintf(stderr, "pipeline residency authority relocation result=%d\n",
                 result);
    return 630 + result;
  }
  if (const int result = CheckAuthorityViews(); result != 0) {
    std::fprintf(stderr, "pipeline residency authority views result=%d\n",
                 result);
    return 720 + result;
  }
  if (const int result = CheckAuthorityPrefetch(); result != 0) {
    std::fprintf(stderr, "pipeline residency authority prefetch result=%d\n",
                 result);
    return 670 + result;
  }
  if (const int result = CheckCycle(); result != 0) {
    std::fprintf(stderr, "pipeline residency cycle result=%d\n", result);
    return 600 + result;
  }
  if (const int result = CheckExecution(); result != 0) {
    std::fprintf(stderr, "pipeline residency execution result=%d\n", result);
    return 700 + result;
  }
  if (const int result = CheckFetchFill(); result != 0) {
    std::fprintf(stderr, "pipeline residency Fetch fill result=%d\n", result);
    return 780 + result;
  }
  if (const int result = CheckSlidingModel(); result != 0) {
    std::fprintf(stderr, "pipeline residency sliding model result=%d\n",
                 result);
    return 800 + result;
  }
  if (const int result = CheckSlidingAuthority(); result != 0) {
    std::fprintf(stderr, "pipeline residency sliding authority result=%d\n",
                 result);
    return 820 + result;
  }
  if (const int result = CheckSlidingGeneration(); result != 0) {
    std::fprintf(stderr, "pipeline residency sliding generation result=%d\n",
                 result);
    return 840 + result;
  }
  if (const int result = CheckNativeSliding(); result != 0) {
    std::fprintf(stderr, "pipeline residency native sliding result=%d\n",
                 result);
    return 850 + result;
  }
  if (const int result = CheckPersistentSliding(); result != 0) {
    std::fprintf(stderr, "pipeline residency persistent sliding result=%d\n",
                 result);
    return 875 + result;
  }
  if (const int result = CheckServiceFreeDirect(); result != 0) {
    std::fprintf(stderr, "pipeline residency service-free direct result=%d\n",
                 result);
    return 890 + result;
  }
  if (const int result = CheckDeviceVsm(); result != 0) {
    std::fprintf(stderr, "pipeline residency device VSM result=%d\n", result);
    return 895 + result;
  }
  if (const int result = CheckWindow(); result != 0) {
    std::fprintf(stderr, "pipeline residency window result=%d\n", result);
    return 900 + result;
  }
  if (const int result = CheckPoolLending(); result != 0) {
    std::fprintf(stderr, "pipeline residency pool result=%d\n", result);
    return 1000 + result;
  }
  if (const int result = CheckPhysicalBufferViews(); result != 0) {
    std::fprintf(stderr, "pipeline residency physical view result=%d\n",
                 result);
    return 1010 + result;
  }
  if (const int result = CheckPoolExtentAdmission(); result != 0) {
    std::fprintf(stderr,
                 "pipeline residency pool extent admission result=%d\n",
                 result);
    return 1020 + result;
  }
  return 0;
}
