#include "backend/local.hpp"

int RunComputeBackendContract() {
  using namespace rund_node_backend_contract;
  if (!CheckStatsProjection()) {
    return 1;
  }
  if (const int map = CheckMapParity(); map != 0) {
    return map;
  }
  if (!CheckDomains()) {
    return 15;
  }
  if (!CheckPrimitiveDomains()) {
    return 16;
  }
  if (!CheckSegments()) {
    return 17;
  }
  if (!CheckMovementDomains()) {
    return 18;
  }
  if (!CheckPartialWriteReset()) {
    return 23;
  }
  if (!CheckExactMovement()) {
    return 20;
  }
  if (!CheckMatrices()) {
    return 21;
  }
  if (!CheckHistogram()) {
    return 22;
  }
  if (!CheckArgsorts()) {
    return 19;
  }
  return 0;
}
