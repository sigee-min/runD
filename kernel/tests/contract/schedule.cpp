#include "contract/schedule/cases.hpp"

int RunProgramTileContract() {
  if (const int rc = RunProgramTileCapacityContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunProgramTileWeightedContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunProgramTileDispatchContract(); rc != 0) {
    return rc;
  }
  return 0;
}
