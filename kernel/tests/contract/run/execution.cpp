#include "contract/cases.hpp"

int main() {
  if (const int rc = RunDispatchContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunProgramTileContract(); rc != 0) {
    return rc;
  }
  return RunWorkspacePlacementContract();
}
