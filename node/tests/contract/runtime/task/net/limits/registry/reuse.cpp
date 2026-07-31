#include "src/host/net/test/view.hpp"
#include "src/host/net/test/ticket.hpp"
#include "src/host/net/test/socket.hpp"
#include "../local.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/api.hpp>

namespace rund::node::test_contract {

bool NetSocketRegistryFdReuseAdmissionFailsClosed() {
  net_limits::SocketPairCleanup reuse_old{};
  NET_LIMIT_ASSERT(net_limits::MakeSocketPair(reuse_old));
  rund::net::Socket old_socket = rund::node::test::net::admit(reuse_old.left);
  NET_LIMIT_ASSERT(rund::node::test::net::generation(old_socket) != 0u);
  const int reused_fd = rund::node::test::net::native(old_socket);
  static_cast<void>(::close(reused_fd));

  net_limits::SocketPairCleanup reuse_new{};
  NET_LIMIT_ASSERT(net_limits::MakeSocketPair(reuse_new));
  if (reuse_new.left != reused_fd) {
    NET_LIMIT_ASSERT(::dup2(reuse_new.left, reused_fd) == reused_fd);
    static_cast<void>(::close(reuse_new.left));
    reuse_new.left = reused_fd;
  }

  rund::SessionConfig options = net_limits::Config();
  options.scheduler.net_socket_registry_capacity = 0u;

  rund::Session runtime{};
  NET_LIMIT_ASSERT(runtime.open(options).ok());

  rund::net::Socket failed_reuse{};
  std::uint64_t generation_after_failed_reuse = 0u;
  rund::net::SendResult inactive_send{};
  rund::net::SendResult stale_send{};
  rund::task::Status joined{};

  const rund::Session::Result report = runtime.scope([&] {
    const rund::task::Handle task =
        rund::task::spawn("net-limit-fd-reuse-admission-fails", [&] {
          failed_reuse = rund::node::test::net::admit(reuse_new.left);
          generation_after_failed_reuse =
              rund::node::test::net::generation(reused_fd);
          const rund::net::SocketView inactive = rund::node::test::net::view(
              reused_fd, generation_after_failed_reuse);
          const std::array<std::byte, 1u> payload{std::byte{'x'}};
          inactive_send = rund::net::send(
              rund::node::test::net::ticket(
                  inactive, rund::net::ready::Interest::Writable),
              std::span<const std::byte>{payload});
          const rund::net::SocketView stale = rund::node::test::net::view(
              reused_fd, rund::node::test::net::generation(old_socket));
          stale_send =
              rund::net::send(rund::node::test::net::ticket(
                                  stale, rund::net::ready::Interest::Writable),
                              std::span<const std::byte>{payload});
        });
    joined = rund::task::join(task);
  });

  NET_LIMIT_ASSERT(runtime.close().ok());
  NET_LIMIT_ASSERT(report.ok());
  NET_LIMIT_ASSERT(joined.ok());
  NET_LIMIT_ASSERT(!failed_reuse);
  NET_LIMIT_ASSERT(generation_after_failed_reuse == 0u);
  NET_LIMIT_ASSERT(!inactive_send.ok());
  NET_LIMIT_ASSERT(inactive_send.code() == rund::ReasonCode::IoFdInvalid);
  NET_LIMIT_ASSERT(!stale_send.ok());
  NET_LIMIT_ASSERT(stale_send.code() == rund::ReasonCode::IoFdInvalid);
  NET_LIMIT_ASSERT(report.tasks().network().admission_rejections() == 1u);
  return true;
}

} // namespace rund::node::test_contract
