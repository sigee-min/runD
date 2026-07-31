#include <rund/net.hpp>
#include <rund/task.hpp>

#include <array>
#include <cstddef>
#include <utility>

rund::task::Task<rund::net::server::PeerResult>
Handle(rund::net::server::Peer peer) {
  std::array<std::byte, 4096u> bytes{};
  const rund::net::ReceiveResult received =
      co_await rund::net::receive(peer.socket.view(), bytes);
  if (!received) {
    co_return rund::net::server::PeerResult::fail(received.code());
  }
  co_return rund::net::server::PeerResult::complete();
}

rund::task::Task<rund::net::server::Result>
Serve(const rund::net::SocketView listener) {
  co_return co_await rund::net::server::serve(
      rund::net::server::Options{
          .listener = listener,
          .accepts = {.max_accepts = 64u},
      },
      [](rund::net::server::Peer peer) {
        return Handle(std::move(peer));
      });
}
