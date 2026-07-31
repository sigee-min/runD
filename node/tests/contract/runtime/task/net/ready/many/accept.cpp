#include "test/assert.hpp"

#include "accept/local.hpp"

int RunRuntimeTaskNetReadyManyAcceptContract() {
  TEST_ASSERT(RunReadyManyAcceptBlockingCase() == 0);
  TEST_ASSERT(RunReadyManyAcceptBoundaryCase() == 0);
  TEST_ASSERT(RunReadyManyAcceptCloseCase() == 0);
  return 0;
}
