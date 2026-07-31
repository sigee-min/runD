#pragma once

#include "test/assert.hpp"

#include <rund/net/ready/many.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#define READY_SET_ASSERT(condition)                                            \
  do {                                                                         \
    if (!(condition)) {                                                        \
      return false;                                                            \
    }                                                                          \
  } while (false)

namespace rund::node::test_contract::ready_sets {

struct SocketPairCleanup {
  int left = -1;
  int right = -1;

  ~SocketPairCleanup();
  SocketPairCleanup() = default;
  SocketPairCleanup(const SocketPairCleanup &) = delete;
  SocketPairCleanup &operator=(const SocketPairCleanup &) = delete;
};

[[nodiscard]] bool MakeSocketPair(SocketPairCleanup &cleanup);
[[nodiscard]] rund::SessionConfig
RunSpec(std::uint32_t task_capacity = 8u, std::uint32_t timer_capacity = 8u,
        std::uint32_t reactor_wait_capacity = 64u) noexcept;
[[nodiscard]] rund::SessionConfig
Config(std::uint32_t task_capacity, std::uint32_t timer_capacity,
       std::uint32_t reactor_wait_capacity,
       std::uint32_t net_ready_set_capacity,
       std::uint32_t net_ready_set_member_capacity) noexcept;
[[nodiscard]] rund::net::ready::Request
ReadableRequest(rund::net::SocketView socket) noexcept;
[[nodiscard]] bool PrepareSocketPair(SocketPairCleanup &cleanup,
                                     rund::net::Socket &reader,
                                     rund::net::Socket &writer);

} // namespace rund::node::test_contract::ready_sets

namespace rund::node::test_contract {

bool NetReadySetCreateAddWaitDestroy();
bool NetReadySetParkWake();
bool NetReadySetsWakeInCausalOrder();
bool NetReadySetRejectsDuplicatesAndStaleSockets();
bool NetReadySetRejectsDuplicateMember();
bool NetReadySetRejectsStaleWaitMember();
bool NetReadySetClearCancelsActiveWait();
bool NetReadySetCapacityFailsClosed();
bool NetReadySetSetCapacityFailsClosed();
bool NetReadySetMemberCapacityFailsClosed();
bool NetReadySetRejectsZeroMaxMembers();
bool NetReadySetClearResetsInsertionIndex();
bool NetReadySetRemovalHoleKeepsInsertionIndex();
bool NetReadySetWaitUsesMembershipSnapshot();
bool NetReadySetChurnStoragePlateaus();
bool NetReadySetStaleMemberRemoveAndWaitFailClosed();
bool NetReadySetRejectsStaleMemberRemove();
bool NetReadySetRejectsStaleMemberWait();

} // namespace rund::node::test_contract
