#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/limits.hpp>
#include <rund/net/ready/set.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool LimitsReportActiveState() {
  net_limits::SocketPairCleanup cleanup{};
  NET_LIMIT_ASSERT(net_limits::MakeSocketPair(cleanup));
  rund::net::Socket socket = rund::node::test::net::admit(cleanup.left);
  NET_LIMIT_ASSERT(rund::node::test::net::generation(socket) != 0u);
  NET_LIMIT_ASSERT(rund::net::nonblocking(socket.view(), true).ok());

  const rund::net::Limits missing = rund::net::limits();
  NET_LIMIT_ASSERT(!missing.ok());
  NET_LIMIT_ASSERT(missing.code() == rund::ReasonCode::NodeRuntimeMissing);

  rund::SessionConfig options = net_limits::Config();
  options.scheduler.net_ready_set_capacity = 3u;
  options.scheduler.net_ready_set_member_capacity = 5u;
  options.scheduler.net_iov_capacity = 7u;
  options.scheduler.net_datagram_capacity_bytes = 11u;
  options.scheduler.net_socket_registry_capacity = 13u;

  rund::Session runtime{};
  NET_LIMIT_ASSERT(runtime.open(options).ok());

  rund::net::ready::Status created{};
  rund::net::ready::Status added{};
  rund::net::Limits limits{};
  rund::net::ready::Status destroyed{};
  rund::task::Status joined{};

  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task = rund::task::spawn("net.limits", [&] {
      created =
          rund::net::ready::create(rund::net::ready::Config{.max_members = 4u});
      if (!created.ok()) {
        return;
      }
      added = rund::net::ready::add(created.set,
                                    net_limits::ReadableRequest(socket.view()));
      limits = rund::net::limits();
      destroyed = rund::net::ready::destroy(created.set);
    });
    joined = rund::task::join(task);
  });

  NET_LIMIT_ASSERT(runtime.close().ok());
  NET_LIMIT_ASSERT(report.ok());
  NET_LIMIT_ASSERT(joined.ok());
  NET_LIMIT_ASSERT(created.ok());
  NET_LIMIT_ASSERT(added.ok());
  NET_LIMIT_ASSERT(limits.ok());
  NET_LIMIT_ASSERT(limits.ready_sets == 1u);
  NET_LIMIT_ASSERT(limits.ready_set_members == 1u);
  NET_LIMIT_ASSERT(limits.max_ready_sets == 3u);
  NET_LIMIT_ASSERT(limits.max_ready_set_members == 5u);
  NET_LIMIT_ASSERT(limits.max_iov == 7u);
  NET_LIMIT_ASSERT(limits.max_datagram_bytes == 11u);
  NET_LIMIT_ASSERT(limits.max_socket_registry_entries == 13u);
  NET_LIMIT_ASSERT(destroyed.ok());
  return true;
}

} // namespace rund::node::test_contract
