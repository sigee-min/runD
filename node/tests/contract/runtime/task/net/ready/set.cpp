#include "sets/local.hpp"

int RunRuntimeTaskNetReadySetContract() {
  TEST_ASSERT(rund::node::test_contract::NetReadySetCreateAddWaitDestroy());
  TEST_ASSERT(rund::node::test_contract::NetReadySetParkWake());
  TEST_ASSERT(rund::node::test_contract::NetReadySetsWakeInCausalOrder());
  TEST_ASSERT(
      rund::node::test_contract::NetReadySetRejectsDuplicatesAndStaleSockets());
  TEST_ASSERT(rund::node::test_contract::NetReadySetClearCancelsActiveWait());
  TEST_ASSERT(rund::node::test_contract::NetReadySetCapacityFailsClosed());
  TEST_ASSERT(rund::node::test_contract::NetReadySetRejectsZeroMaxMembers());
  TEST_ASSERT(
      rund::node::test_contract::NetReadySetClearResetsInsertionIndex());
  TEST_ASSERT(
      rund::node::test_contract::NetReadySetRemovalHoleKeepsInsertionIndex());
  TEST_ASSERT(
      rund::node::test_contract::NetReadySetWaitUsesMembershipSnapshot());
  TEST_ASSERT(rund::node::test_contract::NetReadySetChurnStoragePlateaus());
  TEST_ASSERT(
      rund::node::test_contract::NetReadySetIdentityTransitionsAreClosed());
  TEST_ASSERT(rund::node::test_contract::NetReadySetCapabilitiesDoNotAlias());
  TEST_ASSERT(rund::node::test_contract::NetReadySetTelemetryHasOneAuthority());
  TEST_ASSERT(rund::node::test_contract::
                  NetReadySetStaleMemberRemoveAndWaitFailClosed());
  return 0;
}
