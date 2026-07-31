#include "../local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetSocketRegistryExternalCloseReleasesCapacity() {
  rund::SessionConfig options = net_limits::Config();
  options.scheduler.net_socket_registry_capacity = 1u;

  rund::Session runtime{};
  NET_LIMIT_ASSERT(runtime.open(options).ok());

  rund::net::OpenResult first{};
  rund::net::CloseResult closed{};
  rund::net::OpenResult second{};
  rund::net::CloseResult second_closed{};
  rund::task::Status joined{};

  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("net-limit-cross-thread-close-release", [&] {
          first = rund::net::open(
              rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                                     .transport = rund::net::Transport::Stream,
                                     .nonblocking = true});
          if (!first.ok()) {
            return;
          }
          std::thread closer{[&] { closed = first.socket.close(); }};
          closer.join();
          second = rund::net::open(
              rund::net::OpenOptions{.family = rund::net::Family::IPv4,
                                     .transport = rund::net::Transport::Stream,
                                     .nonblocking = true});
          if (second.ok()) {
            second_closed = second.socket.close();
          }
        });
    joined = rund::task::join(task);
  });

  NET_LIMIT_ASSERT(runtime.close().ok());
  NET_LIMIT_ASSERT(report.ok());
  NET_LIMIT_ASSERT(joined.ok());
  NET_LIMIT_ASSERT(first.ok());
  NET_LIMIT_ASSERT(closed.ok());
  NET_LIMIT_ASSERT(second.ok());
  NET_LIMIT_ASSERT(second_closed.ok());
  NET_LIMIT_ASSERT(report.tasks().network().admission_rejections() == 0u);
  return true;
}

} // namespace rund::node::test_contract
