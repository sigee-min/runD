#include "local.hpp"

#include <rund/net/bytes.hpp>
#include <rund/net/connection.hpp>
#include <rund/net/io.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/ready.hpp>
#include <rund/net/server/options.hpp>
#include <rund/net/server/peer.hpp>
#include <rund/net/server/serve.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "test/assert.hpp"

#include <array>
#include <span>
#include <utility>

namespace server_inline_detail {
namespace {

struct PublicClient final {
  rund::net::connect::Result started{};
  rund::net::connect::Result connected{};
  rund::net::SendResult sent{};
};

[[nodiscard]] rund::task::Task<void>
SendPublicByte(const rund::net::SocketView socket,
               const rund::net::Address address,
               const std::span<const std::byte> bytes, PublicClient &result) {
  result.started = rund::net::connect::start(socket, address);
  if (!result.started) {
    co_return;
  }
  auto writable = co_await rund::net::ready::write(socket);
  if (!writable.ready()) {
    result.connected = rund::net::connect::Result{writable.code()};
    co_return;
  }
  result.connected = rund::net::connect::finish(std::move(writable), address);
  if (!result.connected) {
    co_return;
  }
  result.sent = co_await rund::net::send(socket, bytes);
}

[[nodiscard]] rund::task::Task<rund::net::server::PeerResult>
ReceivePublicByte(rund::net::server::Peer peer, std::byte &value) {
  std::array<std::byte, 1u> bytes{};
  const rund::net::ReceiveResult received =
      co_await rund::net::receive(peer.socket.view(), bytes);
  if (!received || received.bytes != 1) {
    co_return rund::net::server::PeerResult::fail(
        received ? rund::ReasonCode::NetPeerHandlerFailed : received.code());
  }
  value = bytes[0];
  co_return rund::net::server::PeerResult::complete();
}

} // namespace

[[nodiscard]] int RunServerPublicIoCase() {
  auto opened_listener = rund::net::open();
  TEST_ASSERT(opened_listener.ok());
  rund::net::Socket listener = std::move(opened_listener.socket);
  TEST_ASSERT(rund::net::bind(listener.view(), rund::net::Address::loopback(
                                                   rund::net::Family::IPv4))
                  .ok());
  TEST_ASSERT(rund::net::listen(listener.view(), 1).ok());
  const auto local = rund::net::local(listener.view());
  TEST_ASSERT(local.ok());

  auto opened_client = rund::net::open();
  TEST_ASSERT(opened_client.ok());
  rund::net::Socket client = std::move(opened_client.socket);
  const std::array payload{std::byte{0x2au}};
  PublicClient client_result{};
  std::byte peer_byte{};
  rund::net::server::Result served{};
  rund::task::Status client_joined{};
  rund::task::Status server_joined{};

  const rund::Session::Result run = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .reactor_wait_capacity = 4u,
                  .observation_capacity = 32u,
                  .host_event_capacity = 32u,
              },
      },
      [&] {
        auto server = [&]() -> rund::task::Task<void> {
          served = co_await rund::net::server::serve(
              rund::net::server::Options{
                  .listener = listener.view(),
                  .accepts = {.max_accepts = 1u},
                  .task_name = "server.peer",
              },
              [&](rund::net::server::Peer peer) {
                return ReceivePublicByte(std::move(peer), peer_byte);
              });
        };
        const auto server_task = rund::task::spawn("server", server());
        const auto client_task = rund::task::spawn(
            "client", SendPublicByte(client.view(), local.address, payload,
                                     client_result));
        client_joined = rund::task::join(client_task);
        server_joined = rund::task::join(server_task);
      });

  TEST_ASSERT(run.ok());
  TEST_ASSERT(client_joined.ok());
  TEST_ASSERT(server_joined.ok());
  TEST_ASSERT(served.ok());
  TEST_ASSERT(client_result.started.ok());
  TEST_ASSERT(client_result.connected.ok());
  TEST_ASSERT(client_result.sent.ok());
  TEST_ASSERT(client_result.sent.bytes == 1);
  TEST_ASSERT(peer_byte == payload[0]);
  TEST_ASSERT(HasCounts(served, 1u, 0u, 0u));
  TEST_ASSERT(served.budget_exhausted);
  return 0;
}

} // namespace server_inline_detail
