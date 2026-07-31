#include "test/assert.hpp"

#include "local.hpp"

int RunRuntimeTaskNetReadyManyConnectContract() {
  TEST_ASSERT(RunNetReadyManyConnectMultiInterestCase() == 0);
  TEST_ASSERT(RunNetReadyManyConnectGenerationCase() == 0);
  return 0;
}
