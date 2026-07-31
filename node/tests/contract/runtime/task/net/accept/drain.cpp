#include "test/assert.hpp"

#include "drain/local.hpp"

int RunRuntimeTaskNetAcceptDrainContract() {
  TEST_ASSERT(RunAcceptDrainWouldBlockCase() == 0);
  TEST_ASSERT(RunAcceptDrainCallbackStopCase() == 0);
  TEST_ASSERT(RunAcceptDrainBudgetExhaustedCase() == 0);
  TEST_ASSERT(RunAcceptDrainNonWouldBlockFailureCase() == 0);
  return 0;
}
