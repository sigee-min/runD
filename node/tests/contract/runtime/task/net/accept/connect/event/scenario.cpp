#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "local.hpp"

#include <rund/net/connection.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <sys/socket.h>
#include <unistd.h>

AcceptConnectEventCase RunAcceptConnectEventScenario() {
  AcceptConnectEventCase event{};
  int listener_fd = -1;
  sockaddr_in listener_address{};
  event.listener_ok =
      MakeAcceptConnectLoopbackListener(&listener_fd, &listener_address);
  if (!event.listener_ok) {
    return event;
  }

  AcceptConnectSocketCleanup listener_cleanup{listener_fd};
  event.listener = rund::node::test::net::admit(listener_cleanup.fd);
  event.listener_nonblocking =
      rund::net::nonblocking(event.listener.view(), true);

  const int client_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  event.client_ok = client_fd >= 0;
  if (!event.client_ok) {
    return event;
  }

  AcceptConnectSocketCleanup client_cleanup{client_fd};
  event.client = rund::node::test::net::admit(client_cleanup.fd);
  event.client_nonblocking = rund::net::nonblocking(event.client.view(), true);
  event.address = AcceptConnectAddressFromSockaddr(listener_address);
  event.report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_event_capacity = 16u,
              },
      },
      [&] {
        event.empty_accept =
            rund::net::accept::one(rund::node::test::net::ticket(
                event.listener.view(), rund::net::ready::Interest::Readable));
        event.started =
            rund::net::connect::start(event.client.view(), event.address);
        for (int attempt = 0; attempt < 100 && !event.accepted.ok();
             ++attempt) {
          event.accepted = rund::net::accept::one(rund::node::test::net::ticket(
              event.listener.view(), rund::net::ready::Interest::Readable));
          if (!event.accepted.ok()) {
            ::usleep(1000);
          }
        }
        for (int attempt = 0; attempt < 100 && !event.finished.ok();
             ++attempt) {
          event.finished = rund::net::connect::finish(
              rund::node::test::net::ticket(
                  event.client.view(), rund::net::ready::Interest::Writable),
              event.address);
          if (!event.finished.ok()) {
            ::usleep(1000);
          }
        }
      });
  return event;
}
