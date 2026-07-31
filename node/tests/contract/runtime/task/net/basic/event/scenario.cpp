#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include <rund/net/bytes.hpp>
#include <span>

#include <sys/socket.h>

BasicEventCase RunBasicEventScenario() {
  BasicEventCase event{};
  int sockets[2] = {-1, -1};
  event.socketpair_ok = ::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0;
  if (!event.socketpair_ok) {
    return event;
  }

  BasicSocketCleanup left_cleanup{sockets[0]};
  BasicSocketCleanup right_cleanup{sockets[1]};
  event.left = rund::node::test::net::admit(left_cleanup.fd);
  event.right = rund::node::test::net::admit(right_cleanup.fd);
  event.report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_event_capacity = 4u,
              },
      },
      [&] {
        event.send = rund::net::direct::send(
            event.left.view(), std::span<const std::byte>{event.out});
        event.recv = rund::net::direct::receive(event.right.view(),
                                                std::span<std::byte>{event.in});
        event.zero_send = rund::net::direct::send(
            event.left.view(),
            std::span<const std::byte>{event.zero_out.data(), 0u});
        event.zero_recv = rund::net::direct::receive(
            event.right.view(), std::span<std::byte>{event.zero_in.data(), 0u});
      });
  return event;
}
