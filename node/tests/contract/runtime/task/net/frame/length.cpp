#include "test/assert.hpp"

#include "length/local.hpp"

int RunRuntimeTaskNetFrameContract() {
  TEST_ASSERT(RunNetFrameLengthCompileCase() == 0);
  TEST_ASSERT(RunNetFrameLengthBasicCase() == 0);
  TEST_ASSERT(RunNetFrameLengthRejectCase() == 0);
  TEST_ASSERT(RunNetFrameLengthDefaultLimitCase() == 0);
  return 0;
}
