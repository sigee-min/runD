#include "test/assert.hpp"

#include "readiness/local.hpp"

int RunRuntimeTaskNetReadinessContract() {
  TEST_ASSERT(RunNetReadinessBasicCase() == 0);
  TEST_ASSERT(RunNetReadinessInvalidCase() == 0);
  TEST_ASSERT(RunNetReadinessWritableCase() == 0);
  TEST_ASSERT(RunNetReadinessParkedCase() == 0);
  TEST_ASSERT(RunNetReadinessHupCase() == 0);
  TEST_ASSERT(RunNetReadinessZeroCase() == 0);
  return 0;
}
