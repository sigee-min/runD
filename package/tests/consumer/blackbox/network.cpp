#include "model.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace package_blackbox {

#if defined(__unix__) || defined(__APPLE__)
rund::task::Task<void> TransferDatagram(
    const rund::net::SocketView sender, const rund::net::SocketView receiver,
    const rund::net::Address destination, const std::span<const std::byte> out,
    const std::span<std::byte> in, rund::net::datagram::SendResult &sent,
    rund::net::datagram::ReceiveResult &received) {
  sent = co_await rund::net::datagram::send(sender, out, destination);
  if (!sent) {
    co_return;
  }
  received = co_await rund::net::datagram::receive(receiver, in);
}

[[nodiscard]] int CheckNetwork() {
  int sockets[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    return Mismatch("host-socketpair");
  }
  rund::host::io::Fd host_writer = rund::host::io::take_native_fd(sockets[0]);
  rund::host::io::Fd host_reader = rund::host::io::take_native_fd(sockets[1]);
  if (!host_writer || !host_reader) {
    return Mismatch("host-fd-admission");
  }
  std::array<std::byte, 4u> host_out{std::byte{'h'}, std::byte{'o'},
                                     std::byte{'s'}, std::byte{'t'}};
  std::array<std::byte, 4u> host_in{};
  rund::host::io::WriteResult host_written{};
  rund::host::io::ReadResult host_read{};
  const rund::SessionConfig host_config{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 2u,
              .ready_queue_capacity = 2u,
              .host_io_capacity = 1u,
              .host_event_capacity = 4u,
              .host_payload_capacity_bytes = 8u,
          },
  };
  rund::Session host_session{};
  const rund::Session::Status host_opened = host_session.open(host_config);
  if (!host_opened) {
    return host_opened.exit_code();
  }
  const int host_outcome = Finish(host_session, [&]() -> int {
    rund::task::Status host_joined{};
    const rund::replay::Record host_record =
        rund::replay::record(host_session, [&](rund::replay::Context &) {
          const rund::task::Handle task = rund::task::spawn(
              "blackbox-host-io",
              TransferHostIo(host_writer.view(), host_reader.view(), host_out,
                             host_in, host_written, host_read));
          host_joined = rund::task::join(task);
        });
    if (!host_record) {
      return host_record.exit_code();
    }
    if (!host_joined) {
      return host_joined.exit_code();
    }
    if (!host_written) {
      return host_written.exit_code();
    }
    if (!host_read) {
      return host_read.exit_code();
    }
    if (host_in != host_out) {
      return Mismatch("host-io");
    }
    rund::host::io::Fd replay_writer =
        rund::host::io::replay_fd(host_writer.id());
    rund::host::io::Fd replay_reader =
        rund::host::io::replay_fd(host_reader.id());
    std::array<std::byte, 4u> replay_in{};
    rund::host::io::WriteResult replay_written{};
    rund::host::io::ReadResult replay_read{};
    rund::task::Status replay_joined{};
    const rund::replay::Check host_replayed = rund::replay::run(
        host_session, host_record, [&](rund::replay::Context &) {
          const rund::task::Handle task = rund::task::spawn(
              "blackbox-host-io",
              TransferHostIo(replay_writer.view(), replay_reader.view(),
                             host_out, replay_in, replay_written, replay_read));
          replay_joined = rund::task::join(task);
        });
    if (!host_replayed) {
      return host_replayed.exit_code();
    }
    if (!replay_joined) {
      return replay_joined.exit_code();
    }
    if (!replay_written) {
      return replay_written.exit_code();
    }
    if (!replay_read) {
      return replay_read.exit_code();
    }
    if (replay_in != host_out || replay_writer.id() != host_writer.id() ||
        replay_reader.id() != host_reader.id()) {
      return Mismatch("host-replay");
    }
    return 0;
  });
  if (host_outcome != 0) {
    return host_outcome;
  }

  rund::net::OpenResult sender_open =
      rund::net::open({.family = rund::net::Family::IPv4,
                       .transport = rund::net::Transport::Datagram});
  if (!sender_open) {
    return sender_open.exit_code();
  }
  rund::net::Socket sender = std::move(sender_open.socket);
  rund::net::OpenResult receiver_open =
      rund::net::open({.family = rund::net::Family::IPv4,
                       .transport = rund::net::Transport::Datagram});
  if (!receiver_open) {
    return receiver_open.exit_code();
  }
  rund::net::Socket receiver = std::move(receiver_open.socket);
  const rund::net::Address loopback =
      rund::net::Address::loopback(rund::net::Family::IPv4);
  const rund::net::BindResult sender_bound =
      rund::net::bind(sender.view(), loopback);
  if (!sender_bound) {
    return sender_bound.exit_code();
  }
  const rund::net::BindResult receiver_bound =
      rund::net::bind(receiver.view(), loopback);
  if (!receiver_bound) {
    return receiver_bound.exit_code();
  }
  const rund::net::LocalResult destination = rund::net::local(receiver.view());
  if (!destination) {
    return destination.exit_code();
  }
  if (destination.address.port() == 0u) {
    return Mismatch("network-address");
  }

  std::array<std::byte, 4u> out{std::byte{'r'}, std::byte{'u'}, std::byte{'n'},
                                std::byte{'D'}};
  std::array<std::byte, 4u> in{};
  rund::net::datagram::SendResult sent{};
  rund::net::datagram::ReceiveResult received{};
  rund::task::Status network_joined{};
  const rund::Session::Result result = rund::run(
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
        const rund::task::Handle task = rund::task::spawn(
            "blackbox-net",
            TransferDatagram(sender.view(), receiver.view(),
                             destination.address,
                             std::span<const std::byte>{out},
                             std::span<std::byte>{in}, sent, received));
        network_joined = rund::task::join(task);
      });
  if (!result) {
    return result.exit_code();
  }
  if (!network_joined) {
    return network_joined.exit_code();
  }
  if (!sent) {
    return sent.exit_code();
  }
  if (!received) {
    return received.exit_code();
  }
  return sent.bytes == static_cast<std::int64_t>(out.size()) &&
                 received.bytes == static_cast<std::int64_t>(out.size()) &&
                 in == out
             ? 0
             : Mismatch("network-transfer");
}
#else
[[nodiscard]] int CheckNetwork() { return 0; }
#endif

} // namespace package_blackbox
