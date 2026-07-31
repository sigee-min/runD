#include "test/assert.hpp"

#include "parallel/local.hpp"

int RunRuntimeTaskNetServerParallelContract() {
  TEST_ASSERT(RunServerParallelSuccessCase() == 0);
  TEST_ASSERT(RunServerParallelFrameCapacityCase() == 0);
  TEST_ASSERT(RunServerParallelHandlerFailureCase() == 0);
  return 0;
}
