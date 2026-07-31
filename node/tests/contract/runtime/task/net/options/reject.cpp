#include "local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/options.hpp>
#include <rund/task/api.hpp>

#include <sys/socket.h>

bool NetSocketOptionsRejectInvalidValues() {
  NetOptionsSocketCloseGuard tcp{};
  NetOptionsSocketCloseGuard udp{};
  NET_OPTIONS_ASSERT(
      OpenNetOptionsInetSocket(rund::net::Transport::Stream, tcp));
  NET_OPTIONS_ASSERT(
      OpenNetOptionsInetSocket(rund::net::Transport::Datagram, udp));

  const rund::net::option::Result negative_recv = rund::net::option::set(
      tcp.socket.view(), rund::net::option::Name::ReceiveBufferBytes,
      rund::net::option::Value{.bytes = -1});
  NET_OPTIONS_ASSERT(!negative_recv.ok());
  NET_OPTIONS_ASSERT(negative_recv.code() == rund::ReasonCode::TaskInvalid);

  const rund::net::option::Result negative_send = rund::net::option::set(
      tcp.socket.view(), rund::net::option::Name::SendBufferBytes,
      rund::net::option::Value{.bytes = -1});
  NET_OPTIONS_ASSERT(!negative_send.ok());
  NET_OPTIONS_ASSERT(negative_send.code() == rund::ReasonCode::TaskInvalid);

  const rund::net::option::Result udp_tcp_nodelay = rund::net::option::set(
      udp.socket.view(), rund::net::option::Name::TcpNoDelay,
      rund::net::option::Value{.flag = true});
  NET_OPTIONS_ASSERT(IsTcpNoDelayUdpAcceptedFailure(udp_tcp_nodelay));
  return true;
}

bool NetSocketOptionsRejectInvalidOptionIdsBeforeEvents() {
  NetOptionsSocketCloseGuard guard{};
  NET_OPTIONS_ASSERT(
      OpenNetOptionsInetSocket(rund::net::Transport::Stream, guard));

  rund::net::option::Result invalid_set{};
  rund::net::option::Result invalid_get{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 4u,
                  .host_event_capacity = 8u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("net-options-invalid-id", [&] {
              invalid_set = rund::net::option::set(
                  guard.socket.view(),
                  static_cast<rund::net::option::Name>(255u),
                  rund::net::option::Value{.flag = true});
              invalid_get = rund::net::option::get(
                  guard.socket.view(),
                  static_cast<rund::net::option::Name>(255u));
            });
        joined = rund::task::join(task);
      });

  NET_OPTIONS_ASSERT(report.ok());
  NET_OPTIONS_ASSERT(joined.ok());
  NET_OPTIONS_ASSERT(!invalid_set.ok());
  NET_OPTIONS_ASSERT(invalid_set.code() == rund::ReasonCode::TaskInvalid);
  NET_OPTIONS_ASSERT(!invalid_get.ok());
  NET_OPTIONS_ASSERT(invalid_get.code() == rund::ReasonCode::TaskInvalid);
  NET_OPTIONS_ASSERT(report.events().empty());
  return true;
}
