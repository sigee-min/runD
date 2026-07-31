#include "test/assert.hpp"

#include "flow/local.hpp"

int RunRuntimeTaskNetFlowContract() {
  TEST_ASSERT(RunNetFlowCompileCase() == 0);
  TEST_ASSERT(RunNetFlowDefaultsCase() == 0);
  TEST_ASSERT(RunNetFlowReserveCase() == 0);
  TEST_ASSERT(RunNetFlowReleaseCase() == 0);
  TEST_ASSERT(RunNetFlowRejectCase() == 0);
  return 0;
}
