#include "many/local.hpp"

#include "test/assert.hpp"

int RunNetCancellationReadyManyCase() {
  const NetCancellationManyCase many = RunNetCancellationManyScenario();
  TEST_ASSERT(many.left_pair_ok);
  TEST_ASSERT(many.right_pair_ok);
  TEST_ASSERT(many.setup_ok);
  TEST_ASSERT(many.report.ok());
  TEST_ASSERT(NetCancellationManyCancelMatches(many));
  TEST_ASSERT(NetCancellationManyCleanupMatches(many));
  TEST_ASSERT(NetCancellationManyCountersMatch(many));
  return 0;
}
