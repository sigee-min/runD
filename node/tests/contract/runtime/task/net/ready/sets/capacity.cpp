#include "local.hpp"

namespace rund::node::test_contract {

bool NetReadySetCapacityFailsClosed() {
  READY_SET_ASSERT(NetReadySetSetCapacityFailsClosed());
  READY_SET_ASSERT(NetReadySetMemberCapacityFailsClosed());
  return true;
}

}  // namespace rund::node::test_contract
