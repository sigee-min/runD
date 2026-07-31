#include "bounded/local.hpp"

int RunComputeBoundedParityContract() {
  if (const int cpu = rund_node_bounded_contract::CheckCpu(); cpu != 0) {
    return cpu;
  }
  if (const int parity = rund_node_bounded_contract::CheckParity();
      parity != 0) {
    return parity;
  }
  return rund_node_bounded_contract::CheckBackends();
}
