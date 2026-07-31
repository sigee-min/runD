#include "test/assert.hpp"

#include "local.hpp"

int RunRuntimeTaskNetReadyManyWriteContract() {
  TEST_ASSERT(RunNetReadyManyWriteReadWriteCase() == 0);
  TEST_ASSERT(RunNetReadyManyWriteImmediateOnlyCase() == 0);
  return 0;
}
