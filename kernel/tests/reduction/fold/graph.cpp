#include "cases.hpp"
#include "graph/cases.hpp"

int RunFoldGraphContract() {
  if (const int rc = RunFoldGraphTopologyContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunFoldGraphValidationContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunFoldGraphReductionContract(); rc != 0) {
    return rc;
  }
  return RunFoldGraphCustomContract();
}
