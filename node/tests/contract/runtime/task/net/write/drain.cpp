#include "test/assert.hpp"

#include "drain/local.hpp"

int RunRuntimeTaskNetWriteDrainContract() {
  TEST_ASSERT(RunWriteDrainSuccessCase() == 0);
  TEST_ASSERT(RunWriteDrainBoundCase() == 0);
  TEST_ASSERT(RunWriteDrainCallbackCase() == 0);
  return 0;
}
