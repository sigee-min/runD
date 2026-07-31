#include "test/assert.hpp"

#include "cancellation/local.hpp"

int RunRuntimeTaskNetCancellationContract() {
  TEST_ASSERT(RunNetCancellationReadableWakeCase() == 0);
  TEST_ASSERT(RunNetCancellationCleanupCase() == 0);
  TEST_ASSERT(RunNetCancellationWritableCase() == 0);
  TEST_ASSERT(RunNetCancellationReadyManyCase() == 0);
  return 0;
}
