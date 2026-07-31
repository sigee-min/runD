#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/bytes.hpp>
#include <rund/net/datagram.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/net/vectored.hpp>
#include <rund/task/api.hpp>

#include <utility>

namespace rund::node::test_contract {

bool NetIovAndDatagramLimitsFailClosed() {
  net_limits::SocketPairCleanup stream_cleanup{};
  NET_LIMIT_ASSERT(net_limits::MakeSocketPair(stream_cleanup));
  rund::net::Socket stream_left =
      rund::node::test::net::admit(stream_cleanup.left);
  NET_LIMIT_ASSERT(rund::node::test::net::generation(stream_left) != 0u);
  NET_LIMIT_ASSERT(rund::net::nonblocking(stream_left.view(), true).ok());

  rund::SessionConfig options = net_limits::Config();
  options.scheduler.net_iov_capacity = 1u;
  options.scheduler.net_datagram_capacity_bytes = 4u;

  rund::Session runtime{};
  NET_LIMIT_ASSERT(runtime.open(options).ok());

  std::array<std::byte, 1u> first{std::byte{'a'}};
  std::array<std::byte, 1u> second{std::byte{'b'}};
  const std::array<rund::net::batch::Slice, 2u> slices{
      rund::net::batch::Slice{.data = first.data(), .size = first.size()},
      rund::net::batch::Slice{.data = second.data(), .size = second.size()},
  };
  std::array<std::byte, 5u> datagram{std::byte{'h'}, std::byte{'e'},
                                     std::byte{'l'}, std::byte{'l'},
                                     std::byte{'o'}};
  rund::net::SendResult vectored{};
  rund::net::datagram::SendResult udp{};
  rund::task::Status joined{};

  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("net-limit-iov-datagram", [&] {
          rund::net::OpenResult opened = rund::net::open(rund::net::OpenOptions{
              .family = rund::net::Family::IPv4,
              .transport = rund::net::Transport::Datagram,
              .nonblocking = true});
          if (!opened.ok()) {
            return;
          }
          net_limits::SocketCloseGuard udp_guard{std::move(opened.socket)};
          vectored = rund::net::batch::send(
              rund::node::test::net::ticket(
                  stream_left.view(), rund::net::ready::Interest::Writable),
              slices);
          udp = rund::net::datagram::send(
              rund::node::test::net::ticket(
                  udp_guard.socket.view(),
                  rund::net::ready::Interest::Writable),
              std::span<const std::byte>{datagram},
              net_limits::LoopbackAnyPort());
        });
    joined = rund::task::join(task);
  });

  NET_LIMIT_ASSERT(runtime.close().ok());
  NET_LIMIT_ASSERT(report.ok());
  NET_LIMIT_ASSERT(joined.ok());
  NET_LIMIT_ASSERT(!vectored.ok());
  NET_LIMIT_ASSERT(vectored.code() == rund::ReasonCode::TaskInvalid);
  NET_LIMIT_ASSERT(!udp.ok());
  NET_LIMIT_ASSERT(udp.code() == rund::ReasonCode::TaskInvalid);
  return true;
}

} // namespace rund::node::test_contract
