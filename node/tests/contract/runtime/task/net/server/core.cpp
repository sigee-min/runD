#include "test/assert.hpp"

#include "core/local.hpp"

int RunRuntimeTaskNetServerContract() {
  TEST_ASSERT(RunServerInvalidListenerCase() == 0);
  TEST_ASSERT(RunServerInlineLoopbackCase() == 0);
  return 0;
}
