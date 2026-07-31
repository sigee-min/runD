#include "sync/local.hpp"

#include "test/assert.hpp"

int RunNetBasicSyncCase() {
  TEST_ASSERT(RunNetBasicSyncTransferCase() == 0);
  TEST_ASSERT(RunNetBasicSyncZeroCase() == 0);
  TEST_ASSERT(RunNetBasicSyncInvalidCase() == 0);
  return 0;
}
