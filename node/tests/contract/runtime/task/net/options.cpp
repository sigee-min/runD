#include "test/assert.hpp"

#include "options/local.hpp"

int RunRuntimeTaskNetOptionsContract() {
  TEST_ASSERT(NetSocketOptionsSetAndReadSelectedOptions());
  TEST_ASSERT(NetSocketOptionsRejectInvalidValues());
  TEST_ASSERT(NetSocketOptionsRejectInvalidOptionIdsBeforeEvents());
  TEST_ASSERT(NetSocketOptionsEventsReplayStable());
  return 0;
}
