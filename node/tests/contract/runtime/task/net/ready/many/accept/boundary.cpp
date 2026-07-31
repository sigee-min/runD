#include "boundary/local.hpp"

#include "test/assert.hpp"

int RunReadyManyAcceptBoundaryCase() {
  const ReadyManyBoundaryCase boundary = RunReadyManyBoundaryScenario();
  TEST_ASSERT(boundary.setup_ok);
  TEST_ASSERT(boundary.nonblocking.ok());
  TEST_ASSERT(boundary.report.ok());
  TEST_ASSERT(boundary.joined.ok());
  TEST_ASSERT(ReadyManyBoundaryEmptyMatches(boundary));
  TEST_ASSERT(ReadyManyBoundaryBudgetMatches(boundary));
  TEST_ASSERT(ReadyManyBoundaryRejectMatches(boundary));
  TEST_ASSERT(ReadyManyBoundaryTimeoutMatches(boundary));
  return 0;
}
