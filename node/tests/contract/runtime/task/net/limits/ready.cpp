#include "local.hpp"

namespace rund::node::test_contract {

bool NetReadySetCapacityLimitFailsClosed() {
  NET_LIMIT_ASSERT(NetReadySetSetCapacityLimitFailsClosed());
  return true;
}

bool NetReadySetMemberCapacityLimitFailsClosed() {
  NET_LIMIT_ASSERT(NetReadySetMemberLimitFailsClosed());
  return true;
}

}  // namespace rund::node::test_contract
