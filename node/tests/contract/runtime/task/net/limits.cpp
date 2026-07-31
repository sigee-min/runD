#include "limits/local.hpp"

int RunRuntimeTaskNetLimitsContract() {
  TEST_ASSERT(rund::node::test_contract::NetReadySetCapacityLimitFailsClosed());
  TEST_ASSERT(
      rund::node::test_contract::NetReadySetMemberCapacityLimitFailsClosed());
  TEST_ASSERT(rund::node::test_contract::NetIovAndDatagramLimitsFailClosed());
  TEST_ASSERT(rund::node::test_contract::LimitsReportActiveState());
  TEST_ASSERT(rund::node::test_contract::
                  NetSocketRegistryCapacityFailsClosedInActiveRuntime());
  return 0;
}
