#include "test/assert.hpp"

#include "connect/local.hpp"

int RunRuntimeTaskNetAcceptConnectContract() {
  TEST_ASSERT(RunAcceptConnectBasicCase() == 0);
  TEST_ASSERT(RunAcceptConnectEventCase() == 0);
  TEST_ASSERT(RunAcceptConnectRefusedCase() == 0);
  return 0;
}
