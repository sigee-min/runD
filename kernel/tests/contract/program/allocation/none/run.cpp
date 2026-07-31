#include "contract/program/allocation/none/cases.hpp"

int RunProgramNoAllocationContract() {
  if (const int rc = RunProgramNoAllocationCompileRunContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunProgramNoAllocationBackendStatsContract(); rc != 0) {
    return rc;
  }
  return RunProgramNoAllocationPlanningContract();
}
