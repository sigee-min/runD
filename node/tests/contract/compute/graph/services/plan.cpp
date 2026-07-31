#include "local.hpp"

#include <cstdio>

int RunComputeGraphPlanContract() {
  using namespace rund_node_graph_services;
  const bool generic = ValidResourcePlan();
  const bool boundary = ValidBoundaryPlan();
  const bool birth = ValidMemoryBirth();
  const bool memory = ValidMemoryPlan();
  if (!generic || !boundary || !birth || !memory) {
    std::fprintf(stderr,
                 "graph services plan failure generic=%d boundary=%d birth=%d "
                 "memory=%d\n",
                 generic, boundary, birth, memory);
    return 36;
  }
  return 0;
}
