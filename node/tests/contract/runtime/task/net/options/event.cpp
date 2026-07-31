#include "local.hpp"
#include <rund/net/address.hpp>
#include <rund/net/options.hpp>
#include <rund/task/api.hpp>

#include <rund/replay.hpp>

#include <cstdint>

#include <sys/socket.h>

bool NetSocketOptionsEventsReplayStable() {
  NetOptionsSocketCloseGuard guard{};
  NET_OPTIONS_ASSERT(
      OpenNetOptionsInetSocket(rund::net::Transport::Stream, guard));

  rund::net::option::Result set_reuse{};
  rund::net::option::Result get_reuse{};
  rund::task::Status joined{};
  auto apply = [&](rund::replay::Context &) {
    const rund::task::Handle task =
        rund::task::spawn("net-options-replay", [&] {
          set_reuse = rund::net::option::set(
              guard.socket.view(), rund::net::option::Name::ReuseAddress,
              rund::net::option::Value{.flag = true});
          get_reuse = rund::net::option::get(
              guard.socket.view(), rund::net::option::Name::ReuseAddress);
        });
    joined = rund::task::join(task);
  };

  rund::Session session{};
  NET_OPTIONS_ASSERT(session.open(rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_workers = 1u,
              .task_capacity = 2u,
              .ready_queue_capacity = 4u,
              .host_event_capacity = 8u,
          },
  }));
  const rund::replay::Live live = rund::replay::live(session, apply);
  NET_OPTIONS_ASSERT(live.ok());
  NET_OPTIONS_ASSERT(joined.ok());
  NET_OPTIONS_ASSERT(set_reuse.ok());
  NET_OPTIONS_ASSERT(get_reuse.ok());
  NET_OPTIONS_ASSERT(live.events().size() == 2u);
  NET_OPTIONS_ASSERT(live.events()[0].kind ==
                     rund::host::EventKind::NetSetSocketOption);
  NET_OPTIONS_ASSERT(live.events()[1].kind ==
                     rund::host::EventKind::NetGetSocketOption);
  NET_OPTIONS_ASSERT(
      live.events()[0].offset ==
      static_cast<std::uint64_t>(rund::net::option::Name::ReuseAddress));
  NET_OPTIONS_ASSERT(
      live.events()[1].offset ==
      static_cast<std::uint64_t>(rund::net::option::Name::ReuseAddress));
  NET_OPTIONS_ASSERT(live.events()[0].requested_bytes == 1u);
  NET_OPTIONS_ASSERT(live.events()[1].requested_bytes == 1u);
  NET_OPTIONS_ASSERT(live.events()[0].completed_bytes == 1u);
  NET_OPTIONS_ASSERT(live.events()[1].completed_bytes == 1u);

  const rund::replay::Record recorded = rund::replay::record(session, apply);
  NET_OPTIONS_ASSERT(recorded.ok());
  NET_OPTIONS_ASSERT(recorded.host_event_count() == 2u);
  const rund::replay::Check replayed =
      rund::replay::run(session, recorded, apply);
  NET_OPTIONS_ASSERT(replayed.ok());
  NET_OPTIONS_ASSERT(replayed.actual().has_value());
  NET_OPTIONS_ASSERT(replayed.actual()->host_event_hash() ==
                     recorded.host_event_hash());
  NET_OPTIONS_ASSERT(session.close());
  return true;
}
