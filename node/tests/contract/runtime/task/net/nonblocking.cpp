#include "test/assert.hpp"

#include "nonblocking/local.hpp"

int RunRuntimeTaskNetNonblockingContract() {
  TEST_ASSERT(RunNetNonblockingBasicCase() == 0);
  TEST_ASSERT(RunNetTryPerCallNonblockingCase() == 0);
  TEST_ASSERT(RunNetNonblockingPartialAndNullCase() == 0);
  TEST_ASSERT(RunNetNonblockingPressureCase() == 0);
  TEST_ASSERT(RunNetNonblockingReplayCase() == 0);
  return 0;
}
