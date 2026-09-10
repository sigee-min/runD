#include "modes/local.hpp"

int RunComputeBoundaryModesContract() {
  using namespace rund::node::test_contract::boundary_modes;

  // Collective 255-tail, empty, extrema, and overflow semantics are owned by
  // compute.collective-modes and are intentionally not recompiled here.
  if (!CheckDomainModes()) {
    return 1;
  }
  if (!CheckBoundedModes()) {
    return 2;
  }
  if (!CheckUnsignedModes()) {
    return 3;
  }
  return CheckFixedModes() ? 0 : 4;
}
