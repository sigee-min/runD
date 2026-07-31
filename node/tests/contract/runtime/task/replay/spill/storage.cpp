#include "local/model.hpp"

int RunRuntimeTaskReplaySpillStorageContract() {
  if (const int rc = replay_spill::RunSegmentsContract(); rc != 0) {
    return rc;
  }
  if (const int rc = replay_spill::RunRejectContract(); rc != 0) {
    return rc;
  }
  return replay_spill::RunCacheContract();
}
