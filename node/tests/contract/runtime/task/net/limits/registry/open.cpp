#include "../local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetSocketRegistryOpenCapacityFailsClosed() {
  rund::SessionConfig options = net_limits::Config();
  options.scheduler.net_socket_registry_capacity = 1u;

  rund::Session runtime{};
  NET_LIMIT_ASSERT(runtime.open(options).ok());

  rund::net::OpenResult first{};
  rund::net::OpenResult second{};
  rund::net::CloseResult closed{};
  rund::task::Status joined{};

  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("net-limit-socket-registry-capacity", [&] {
          first = rund::net::open(
              rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                                     .transport = rund::net::Transport::Stream,
                                     .nonblocking = true});
          second = rund::net::open(
              rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                                     .transport = rund::net::Transport::Stream,
                                     .nonblocking = true});
          if (first.ok()) {
            closed = first.socket.close();
          }
        });
    joined = rund::task::join(task);
  });

  NET_LIMIT_ASSERT(runtime.close().ok());
  NET_LIMIT_ASSERT(report.ok());
  NET_LIMIT_ASSERT(joined.ok());
  NET_LIMIT_ASSERT(first.ok());
  NET_LIMIT_ASSERT(!second.ok());
  NET_LIMIT_ASSERT(second.code() == rund::ReasonCode::TaskCapacityExceeded);
  NET_LIMIT_ASSERT(report.tasks().network().admission_rejections() == 1u);
  NET_LIMIT_ASSERT(closed.ok());
  return true;
}

} // namespace rund::node::test_contract
