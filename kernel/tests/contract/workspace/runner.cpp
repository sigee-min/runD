#include "cases.hpp"

namespace {

int Run(int (*test)()) {
  const int result = test();
  return result == 0 ? 0 : result;
}

} // namespace

int RunWorkspacePlacementContract() {
  if (const int result = Run(RunWorkspaceCapacityContract); result != 0) {
    return result;
  }
  if (const int result = Run(RunWorkspaceBalanceContract); result != 0) {
    return result;
  }
  if (const int result = Run(RunWorkspacePacketWorkUnitsContract); result != 0) {
    return result;
  }
  if (const int result = Run(RunWorkspaceWeightedContract); result != 0) {
    return result;
  }
  if (const int result = Run(RunWorkspaceHintedDispatchContract); result != 0) {
    return result;
  }
  return 0;
}
