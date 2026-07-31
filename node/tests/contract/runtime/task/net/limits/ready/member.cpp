#include "src/host/net/test/socket.hpp"
#include "../local.hpp"
#include <rund/net/ready/set.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetReadySetMemberLimitFailsClosed() {
  net_limits::SocketPairCleanup cleanup{};
  NET_LIMIT_ASSERT(net_limits::MakeSocketPair(cleanup));
  rund::net::Socket first_socket = rund::node::test::net::admit(cleanup.left);
  rund::net::Socket second_socket = rund::node::test::net::admit(cleanup.right);
  NET_LIMIT_ASSERT(rund::node::test::net::generation(first_socket) != 0u);
  NET_LIMIT_ASSERT(rund::node::test::net::generation(second_socket) != 0u);
  NET_LIMIT_ASSERT(rund::net::nonblocking(first_socket.view(), true).ok());
  NET_LIMIT_ASSERT(rund::net::nonblocking(second_socket.view(), true).ok());

  rund::SessionConfig options = net_limits::Config();
  options.scheduler.net_ready_set_member_capacity = 1u;

  rund::Session runtime{};
  NET_LIMIT_ASSERT(runtime.open(options).ok());

  rund::net::ready::Status created{};
  rund::net::ready::Status first_member{};
  rund::net::ready::Status second_member{};
  rund::net::ready::Status destroyed{};
  rund::task::Status joined{};

  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("net-limit-ready-set-member-capacity", [&] {
          created = rund::net::ready::create(
              rund::net::ready::Config{.max_members = 2u});
          if (!created.ok()) {
            return;
          }
          first_member = rund::net::ready::add(
              created.set, net_limits::ReadableRequest(first_socket.view()));
          second_member = rund::net::ready::add(
              created.set, net_limits::ReadableRequest(second_socket.view()));
          destroyed = rund::net::ready::destroy(created.set);
        });
    joined = rund::task::join(task);
  });

  NET_LIMIT_ASSERT(runtime.close().ok());
  NET_LIMIT_ASSERT(report.ok());
  NET_LIMIT_ASSERT(joined.ok());
  NET_LIMIT_ASSERT(created.ok());
  NET_LIMIT_ASSERT(first_member.ok());
  NET_LIMIT_ASSERT(!second_member.ok());
  NET_LIMIT_ASSERT(second_member.code() ==
                   rund::ReasonCode::ReactorWaitCapacityExceeded);
  NET_LIMIT_ASSERT(destroyed.ok());
  return true;
}

} // namespace rund::node::test_contract
