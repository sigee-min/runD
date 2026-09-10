#include "plan/local.hpp"

namespace node_accel_contract {

bool CpuSimdFreezesExecutorSelectorsAndScratchBoundary() {
  if (!CheckExecutorSelectorsAndFixedScratch()) {
    return false;
  }
  if (!CheckIntegerScratch()) {
    return false;
  }
  if (!CheckCommitDemand()) {
    return false;
  }
  if (!CheckStablePrefixFailure()) {
    return false;
  }
  return CheckStableEdgeAuthority();
}

} // namespace node_accel_contract
