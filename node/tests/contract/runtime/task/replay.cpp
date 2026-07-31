#include "replay/local.hpp"

int RunRuntimeTaskReplayContract() {
  RuntimeReplayFixture fixture{};
  if (const int rc = MakeRuntimeReplayFixture(fixture); rc != 0)
    return rc;
  if (const int rc = CheckReplayRecordContract(fixture); rc != 0) {
    return rc;
  }
  if (const int rc = CheckReplayDecodeContract(fixture); rc != 0) {
    return rc;
  }
  if (const int rc = CheckReplayDiffContract(fixture); rc != 0) {
    return rc;
  }
  if (const int rc = CheckReplayFailureContract(); rc != 0) {
    return rc;
  }
  if (const int rc = CheckReplayRunContract(fixture); rc != 0) {
    return rc;
  }
  return 0;
}
