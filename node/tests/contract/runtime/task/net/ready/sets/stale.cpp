#include "local.hpp"

namespace rund::node::test_contract {

bool NetReadySetRejectsDuplicatesAndStaleSockets() {
  READY_SET_ASSERT(NetReadySetRejectsDuplicateMember());
  READY_SET_ASSERT(NetReadySetRejectsStaleWaitMember());
  return true;
}

bool NetReadySetStaleMemberRemoveAndWaitFailClosed() {
  READY_SET_ASSERT(NetReadySetRejectsStaleMemberRemove());
  READY_SET_ASSERT(NetReadySetRejectsStaleMemberWait());
  return true;
}

} // namespace rund::node::test_contract
