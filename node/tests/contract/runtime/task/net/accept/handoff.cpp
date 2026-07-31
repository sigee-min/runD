#include "test/assert.hpp"

#include "handoff/local.hpp"

int RunRuntimeTaskNetAcceptHandoffContract() {
  TEST_ASSERT(RunNetAcceptHandoffPreparedSocketCase() == 0);
  return 0;
}
