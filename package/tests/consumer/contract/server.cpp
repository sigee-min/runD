#include <rund/net.hpp>
#include <rund/session.hpp>
#include <rund/task.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

namespace {

struct Client final {
  rund::net::connect::Result started{};
  rund::net::connect::Result connected{};
  rund::net::SendResult sent{};
};

template <class Result>
[[nodiscard]] int Exit(const char *stage, const Result &result) {
  const std::string_view error = result.error();
  std::fprintf(stderr, "server.%s: %.*s\n", stage,
               static_cast<int>(error.size()), error.data());
  return result.exit_code();
}

rund::task::Task<void> Send(const rund::net::SocketView socket,
                            const rund::net::Address address,
                            const std::span<const std::byte> bytes,
                            Client &result) {
  result.started = rund::net::connect::start(socket, address);
  if (!result.started) {
    co_return;
  }
  auto connect_ready = co_await rund::net::ready::write(socket);
  if (!connect_ready.ready()) {
    result.connected = rund::net::connect::Result{connect_ready.code()};
    co_return;
  }
  result.connected =
      rund::net::connect::finish(std::move(connect_ready), address);
  if (!result.connected) {
    co_return;
  }
  result.sent = co_await rund::net::send(socket, bytes);
}

rund::task::Task<rund::net::server::PeerResult>
Receive(rund::net::server::Peer peer, std::byte &value) {
  std::array<std::byte, 1u> bytes{};
  const rund::net::ReceiveResult received =
      co_await rund::net::receive(peer.socket.view(), bytes);
  if (!received) {
    co_return rund::net::server::PeerResult::fail(received.code());
  }
  if (received.bytes != 1) {
    co_return rund::net::server::PeerResult::fail(
        rund::ReasonCode::NetPeerHandlerFailed);
  }
  value = bytes[0];
  co_return rund::net::server::PeerResult::complete();
}

} // namespace

int main() {
  auto opened_listener = rund::net::open();
  if (!opened_listener) {
    return Exit("listener.open", opened_listener);
  }
  rund::net::Socket listener = std::move(opened_listener.socket);

  const auto bound = rund::net::bind(
      listener.view(), rund::net::Address::loopback(rund::net::Family::IPv4));
  if (!bound) {
    return Exit("listener.bind", bound);
  }
  const auto listening = rund::net::listen(listener.view(), 1);
  if (!listening) {
    return Exit("listener.listen", listening);
  }
  const auto address = rund::net::local(listener.view());
  if (!address) {
    return Exit("listener.local", address);
  }

  auto opened_client = rund::net::open();
  if (!opened_client) {
    return Exit("client.open", opened_client);
  }
  rund::net::Socket client = std::move(opened_client.socket);
  const std::array payload{std::byte{0x2au}};
  Client client_result{};
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
                return Receive(std::move(peer), peer_byte);
              });
        };
        const auto server_task = rund::task::spawn("server", server());
        const auto client_task =
            rund::task::spawn("client", Send(client.view(), address.address,
                                             payload, client_result));
        client_joined = rund::task::join(client_task);
        server_joined = rund::task::join(server_task);
      });

  if (!run) {
    std::fprintf(
        stderr,
        "server.state: client_join=%s server_join=%s serve=%s start=%s "
        "connect=%s send=%s spawned=%llu completed=%llu failed=%llu "
        "parked=%llu resumed=%llu reactor_waits=%llu observations=%llu\n",
        rund::ReasonString(client_joined.code()),
        rund::ReasonString(server_joined.code()),
        rund::ReasonString(served.code()),
        rund::ReasonString(client_result.started.code()),
        rund::ReasonString(client_result.connected.code()),
        rund::ReasonString(client_result.sent.code()),
        static_cast<unsigned long long>(run.tasks().spawned()),
        static_cast<unsigned long long>(run.tasks().completed()),
        static_cast<unsigned long long>(run.tasks().failed()),
        static_cast<unsigned long long>(run.tasks().parked()),
        static_cast<unsigned long long>(run.tasks().resumed()),
        static_cast<unsigned long long>(run.tasks().reactor_waits()),
        static_cast<unsigned long long>(run.tasks().observations()));
    std::fprintf(
        stderr,
        "server.scheduler: coroutine_resumes=%llu parks=%llu wakes=%llu "
        "completions=%llu joins=%llu root_fast=%llu root_miss=%llu "
        "completion_spins=%llu spin_hits=%llu spin_misses=%llu "
        "single_wakes=%llu requeued=%llu max_ready=%llu hot_entries=%llu\n",
        static_cast<unsigned long long>(run.tasks().coroutine_resumes()),
        static_cast<unsigned long long>(run.tasks().coroutine_parks()),
        static_cast<unsigned long long>(run.tasks().coroutine_wakes()),
        static_cast<unsigned long long>(run.tasks().coroutine_completions()),
        static_cast<unsigned long long>(run.tasks().joins()),
        static_cast<unsigned long long>(run.tasks().root_join_ready_fast_paths()),
        static_cast<unsigned long long>(
            run.tasks().root_join_ready_fast_path_misses()),
        static_cast<unsigned long long>(
            run.tasks().lane_completion_spin_waits()),
        static_cast<unsigned long long>(
            run.tasks().lane_completion_spin_hits()),
        static_cast<unsigned long long>(
            run.tasks().lane_completion_spin_misses()),
        static_cast<unsigned long long>(
            run.tasks().lane_completion_single_wakes()),
        static_cast<unsigned long long>(
            run.tasks().lane_dispatch_requeued_tasks()),
        static_cast<unsigned long long>(run.tasks().max_ready_depth()),
        static_cast<unsigned long long>(run.tasks().root_hot_path_entries()));
    for (const rund::task::Observation &observation : run.observations()) {
      std::fprintf(stderr,
                   "server.observation: sequence=%llu kind=%u task=%llu "
                   "wait=%llu fd=%d interest=%hd revents=%hd reason=%s\n",
                   static_cast<unsigned long long>(observation.sequence),
                   static_cast<unsigned>(observation.kind),
                   static_cast<unsigned long long>(observation.task_id),
                   static_cast<unsigned long long>(observation.wait_id),
                   observation.fd, observation.interest, observation.revents,
                   rund::ReasonString(observation.reason_code));
    }
    return Exit("session.run", run);
  }
  if (!client_joined) {
    return Exit("client.join", client_joined);
  }
  if (!server_joined) {
    return Exit("server.join", server_joined);
  }
  if (!served) {
    return Exit("server.serve", served);
  }
  if (!client_result.started) {
    return Exit("client.start", client_result.started);
  }
  if (!client_result.connected) {
    return Exit("client.connect", client_result.connected);
  }
  if (!client_result.sent) {
    return Exit("client.send", client_result.sent);
  }
  return client_result.sent.bytes == 1 && peer_byte == payload[0] &&
                 served.accepted == 1u && served.started == 1u &&
                 served.completed == 1u && served.failed == 0u &&
                 served.stopped == 0u && served.budget_exhausted
             ? 0
             : 2;
}
