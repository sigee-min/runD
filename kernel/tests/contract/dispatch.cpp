#include "contract/dispatch/cases.hpp"

int RunDispatchContract() {
  if (const int rc = RunDispatchPartitionContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunDispatchTelemetryContract(); rc != 0) {
    return rc;
  }
  return RunDispatchFailureContract();
}
