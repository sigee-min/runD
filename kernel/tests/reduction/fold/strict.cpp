#include "cases.hpp"
#include "strict/cases.hpp"

int RunStrictFoldContract() {
  if (const int rc = RunStrictFloat32Contract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunStrictFloat64Contract(); rc != 0) {
    return rc;
  }
  return RunStrictOrderedFloatContract();
}
