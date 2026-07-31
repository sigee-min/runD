#include "cancel/local.hpp"

#include "test/assert.hpp"

int NetStatsCloseTimeoutCancellation() {
  const NetStatsCancelCase stats = RunNetStatsCancelScenario();
  TEST_ASSERT(stats.cancel_pair_ok);
  TEST_ASSERT(stats.close_pair_ok);
  TEST_ASSERT(stats.setup_ok);
  TEST_ASSERT(stats.report.ok());
  TEST_ASSERT(stats.joined.ok());
  TEST_ASSERT(NetStatsCancelResultMatches(stats));
  TEST_ASSERT(NetStatsCloseResultMatches(stats));
  TEST_ASSERT(NetStatsCancelCountersMatch(stats));
  return 0;
}
