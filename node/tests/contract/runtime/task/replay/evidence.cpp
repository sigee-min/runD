#include "evidence/local.hpp"

int RunRuntimeTaskReplayEvidenceContract() {
  if (const int rc = RunReplayReactorEvidenceContract(); rc != 0)
    return rc;
  if (const int rc = RunReplayNetEvidenceContract(); rc != 0)
    return rc;
  if (const int rc = RunReplayHostRejectContract(); rc != 0)
    return rc;
  if (const int rc = RunReplayHostCommitContract(); rc != 0)
    return rc;
  return RunReplayHostCapacityContract();
}
