#include "../local.hpp"
#include <rund/net/ready/set.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetReadySetMemberCapacityFailsClosed() {
  std::array<ready_sets::SocketPairCleanup, 2u> cleanup{};
  std::array<rund::net::Socket, 2u> readers{};
  std::array<rund::net::Socket, 2u> writers{};
  for (std::size_t index = 0u; index < cleanup.size(); ++index) {
    READY_SET_ASSERT(ready_sets::PrepareSocketPair(
        cleanup[index], readers[index], writers[index]));
  }

  rund::net::ready::Status member_set{};
  rund::net::ready::Status first_member{};
  rund::net::ready::Status second_member{};
  rund::net::ready::Status destroyed{};
  rund::task::Status joined{};
  rund::Session runtime{};
  const rund::SessionConfig options = ready_sets::Config(4u, 4u, 8u, 2u, 1u);
  READY_SET_ASSERT(runtime.open(options).ok());
  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("net-ready-set-member-capacity", [&] {
          member_set = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 2u});
          if (!member_set.ok()) {
            return;
          }
          first_member = rund::net::ready::add(
              member_set.set, ready_sets::ReadableRequest(readers[0u].view()));
          second_member = rund::net::ready::add(
              member_set.set, ready_sets::ReadableRequest(readers[1u].view()));
          destroyed = rund::net::ready::destroy(member_set.set);
        });
    joined = rund::task::join(task);
  });
  READY_SET_ASSERT(runtime.close().ok());

  READY_SET_ASSERT(report.ok());
  READY_SET_ASSERT(joined.ok());
  READY_SET_ASSERT(member_set.ok());
  READY_SET_ASSERT(first_member.ok());
  READY_SET_ASSERT(!second_member.ok());
  READY_SET_ASSERT(second_member.code() ==
                   rund::ReasonCode::ReactorWaitCapacityExceeded);
  READY_SET_ASSERT(destroyed.ok());
  return true;
}

} // namespace rund::node::test_contract
