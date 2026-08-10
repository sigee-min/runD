#include "local.hpp"

#include <cstdio>

int RunComputeVirtualResidencyOracleContract() {
  const int result = rund_node_test_virtual::CheckVirtualBackingOracle();
  if (result != 0) {
    std::fprintf(stderr, "compute virtual backing oracle result=%d\n", result);
  }
  return result;
}
