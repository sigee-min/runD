#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "../local.hpp"
#include <rund/net/connection.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetSocketRegistryAcceptCapacityClosesAccepted() {
  int listener_fd = -1;
  sockaddr_in listener_address{};
  NET_LIMIT_ASSERT(
      net_limits::MakeLoopbackListener(&listener_fd, &listener_address));
  net_limits::NativeSocketCleanup listener_cleanup{listener_fd};
  rund::net::Socket listener =
      rund::node::test::net::admit(listener_cleanup.fd);
  NET_LIMIT_ASSERT(rund::node::test::net::generation(listener) != 0u);
  NET_LIMIT_ASSERT(rund::net::nonblocking(listener.view(), true).ok());

  const int client_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  NET_LIMIT_ASSERT(client_fd >= 0);
  net_limits::NativeSocketCleanup client_cleanup{client_fd};
  NET_LIMIT_ASSERT(::connect(client_fd,
                             reinterpret_cast<sockaddr *>(&listener_address),
                             sizeof(listener_address)) == 0);

  rund::SessionConfig options = net_limits::Config();
  options.scheduler.net_socket_registry_capacity = 0u;

  rund::Session runtime{};
  NET_LIMIT_ASSERT(runtime.open(options).ok());

  rund::net::accept::Result accepted{};
  rund::task::Status joined{};

  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("net-limit-accept-registry-capacity", [&] {
          accepted = rund::net::accept::one(rund::node::test::net::ticket(
              listener.view(), rund::net::ready::Interest::Readable));
        });
    joined = rund::task::join(task);
  });

  NET_LIMIT_ASSERT(runtime.close().ok());
  NET_LIMIT_ASSERT(report.ok());
  NET_LIMIT_ASSERT(joined.ok());
  NET_LIMIT_ASSERT(!accepted.ok());
  NET_LIMIT_ASSERT(accepted.code() == rund::ReasonCode::TaskCapacityExceeded);
  NET_LIMIT_ASSERT(rund::node::test::net::native(accepted.socket) < 0);
  NET_LIMIT_ASSERT(report.tasks().network().admission_rejections() == 1u);
  return true;
}

} // namespace rund::node::test_contract
