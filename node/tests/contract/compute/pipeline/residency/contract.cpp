#include "local.hpp"

#include <cstdio>

int RunComputePipelineResidencyContract() {
  using namespace rund_node_test_pipeline_residency;
  if (const int result = CheckPlanner(); result != 0) {
    std::fprintf(stderr, "pipeline residency planner result=%d\n", result);
    return 100 + result;
  }
  if (const int result = CheckIdentity(); result != 0) {
    std::fprintf(stderr, "pipeline residency identity result=%d\n", result);
    return 200 + result;
  }
  return 0;
}
