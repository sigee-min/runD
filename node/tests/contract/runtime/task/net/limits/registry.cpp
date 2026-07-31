#include "local.hpp"

namespace rund::node::test_contract {

bool NetSocketRegistryCapacityFailsClosedInActiveRuntime() {
  NET_LIMIT_ASSERT(NetSocketRegistryOpenCapacityFailsClosed());
  NET_LIMIT_ASSERT(NetSocketRegistryExternalCloseReleasesCapacity());
  NET_LIMIT_ASSERT(NetSocketRegistryAcceptCapacityClosesAccepted());
  NET_LIMIT_ASSERT(NetSocketRegistryFdReuseAdmissionFailsClosed());
  return true;
}

}  // namespace rund::node::test_contract
