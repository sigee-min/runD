#include "test/assert.hpp"

#include "readiness/local.hpp"

int RunRuntimeTaskNetTimedReadinessContract() {
  TEST_ASSERT(RunTimedReadinessReadyCase() == 0);
  TEST_ASSERT(RunTimedReadinessCloseCase() == 0);
  TEST_ASSERT(RunTimedReadinessInvalidCase() == 0);
  TEST_ASSERT(RunTimedReadinessCoroutineCase() == 0);
  return 0;
}
