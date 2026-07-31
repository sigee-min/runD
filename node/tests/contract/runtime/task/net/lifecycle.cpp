#include "test/assert.hpp"

#include "lifecycle/local.hpp"

int RunRuntimeTaskNetLifecycleContract() {
  TEST_ASSERT(RunNetLifecycleInvalidCloseCase() == 0);
  TEST_ASSERT(RunNetLifecycleOwnerCase() == 0);
  TEST_ASSERT(RunNetLifecycleLeaseCase() == 0);
  TEST_ASSERT(RunNetLifecycleTicketCase() == 0);
  TEST_ASSERT(RunNetLifecycleCloseInvalidatesWaitCase() == 0);
  TEST_ASSERT(RunNetLifecycleReuseCase() == 0);
  return 0;
}
