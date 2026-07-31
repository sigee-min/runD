#include "test/assert.hpp"

#include "local.hpp"

int RunRuntimeTaskNetReadyManyReadContract() {
  TEST_ASSERT(RunNetReadyManyReadMissingRuntimeCase() == 0);
  TEST_ASSERT(RunNetReadyManyReadImmediateCase() == 0);
  TEST_ASSERT(RunNetReadyManyReadBatchCase() == 0);
  TEST_ASSERT(RunNetReadyManyValidationScaleCase() == 0);
  return 0;
}
