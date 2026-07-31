#include "test/assert.hpp"

#include "suspend/local.hpp"

int RunRuntimeTaskNetTimedReadySuspendContract() {
  TEST_ASSERT(RunTimedSuspendCoroutineCase() == 0);
  return 0;
}
