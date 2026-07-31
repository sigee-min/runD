#include "test/assert.hpp"

#include "many/local.hpp"

int RunRuntimeTaskNetReadyManyContract() {
  TEST_ASSERT(RunRuntimeTaskNetReadyManyReadContract() == 0);
  TEST_ASSERT(RunRuntimeTaskNetReadyManyWriteContract() == 0);
  TEST_ASSERT(RunRuntimeTaskNetReadyManyAcceptContract() == 0);
  TEST_ASSERT(RunRuntimeTaskNetReadyManyConnectContract() == 0);
  return 0;
}
