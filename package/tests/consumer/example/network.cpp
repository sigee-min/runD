#include <rund/net.hpp>
#include <rund/task.hpp>

#include <cstddef>
#include <span>

rund::task::Task<void> Read(const rund::net::SocketView socket,
                            const std::span<std::byte> bytes,
                            rund::net::ReceiveResult &result) {
  result = co_await rund::net::receive(socket, bytes);
}

rund::task::Task<void> SendPacket(const rund::net::SocketView socket,
                                  const std::span<const std::byte> bytes,
                                  const rund::net::Address peer,
                                  rund::net::datagram::SendResult &result) {
  result = co_await rund::net::datagram::send(socket, bytes, peer);
}
