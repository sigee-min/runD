#include "src/host/net/test/socket.hpp"
#include "../local.hpp"

#include "test/assert.hpp"

#include <array>
#include <rund/net/bytes.hpp>
#include <rund/net/socket.hpp>
#include <span>
#include <string_view>

#include <sys/socket.h>
#include <unistd.h>

int RunNetBasicClosedPeerCase() {
  int sockets[2] = {-1, -1};
  TEST_ASSERT(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  BasicSocketCleanup local_cleanup{sockets[0]};
  BasicSocketCleanup remote_cleanup{sockets[1]};
  rund::net::Socket local = rund::node::test::net::admit(local_cleanup.fd);
#if defined(SO_NOSIGPIPE)
  int no_sigpipe = 0;
  socklen_t no_sigpipe_size = sizeof(no_sigpipe);
  TEST_ASSERT(::getsockopt(rund::node::test::net::native(local), SOL_SOCKET,
                           SO_NOSIGPIPE, &no_sigpipe, &no_sigpipe_size) == 0);
  TEST_ASSERT(no_sigpipe == 1);
#endif
  TEST_ASSERT(::close(remote_cleanup.fd) == 0);
  remote_cleanup.release();

  std::array<std::byte, 4> out{std::byte{'p'}, std::byte{'e'}, std::byte{'e'},
                               std::byte{'r'}};
  rund::net::SendResult sent{};
  g_sigpipe_count = 0;
  const sig_atomic_t sigpipe_before = g_sigpipe_count;
  ScopedSigpipeHandler sigpipe_handler{};
  TEST_ASSERT(sigpipe_handler.active);
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_event_capacity = 1u,
              },
      },
      [&] {
        sent = rund::net::direct::send(local.view(),
                                       std::span<const std::byte>{out});
      });
  TEST_ASSERT(g_sigpipe_count == sigpipe_before);
#if defined(SO_NOSIGPIPE)
  no_sigpipe = 0;
  no_sigpipe_size = sizeof(no_sigpipe);
  TEST_ASSERT(::getsockopt(rund::node::test::net::native(local), SOL_SOCKET,
                           SO_NOSIGPIPE, &no_sigpipe, &no_sigpipe_size) == 0);
  TEST_ASSERT(no_sigpipe == 1);
#endif
  TEST_ASSERT(report.ok());
  TEST_ASSERT(!sent.ok());
  TEST_ASSERT(sent.code() == rund::ReasonCode::IoSyscallFailed);
  TEST_ASSERT(std::string_view{sent.error()} == "io_syscall_failed");
  TEST_ASSERT(sent.bytes == -1);
  TEST_ASSERT(sent.native_error != 0);
  TEST_ASSERT(report.events().size() == 1u);
  TEST_ASSERT(report.events()[0].kind == rund::host::EventKind::NetSend);
  TEST_ASSERT(report.events()[0].status == rund::host::Status::SyscallFailed);
  TEST_ASSERT(report.events()[0].host_handle_id != 0u);
  TEST_ASSERT(report.events()[0].requested_bytes == out.size());
  TEST_ASSERT(report.events()[0].completed_bytes == 0u);
  TEST_ASSERT(report.events()[0].native_errno == sent.native_error);
  TEST_ASSERT(report.events()[0].native_errno != 0);
  TEST_ASSERT(report.events()[0].payload_hash.value == 0u);
  TEST_ASSERT(report.tasks().network().bytes_sent() == 0u);
  return 0;
}
