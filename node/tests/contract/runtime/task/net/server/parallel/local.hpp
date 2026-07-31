#pragma once

#include <rund/net/accept.hpp>
#include <rund/net/server/result.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/handle.hpp>

#include <cstddef>
#include <cstdint>

#include <netinet/in.h>

static_assert(sizeof(rund::net::accept::Drain) == 12u);
static_assert(sizeof(rund::task::Handle) == 32u);

inline constexpr std::size_t kServerParallelWarmClientEnvelope = 6u;
inline constexpr int kServerParallelListenerBacklog = 8;
static_assert(kServerParallelListenerBacklog >=
              static_cast<int>(kServerParallelWarmClientEnvelope));

struct ServerParallelSocketCleanup {
  int fd = -1;

  ~ServerParallelSocketCleanup();

  ServerParallelSocketCleanup() = default;
  explicit ServerParallelSocketCleanup(int native) noexcept;
  ServerParallelSocketCleanup(const ServerParallelSocketCleanup &) = delete;
  ServerParallelSocketCleanup &
  operator=(const ServerParallelSocketCleanup &) = delete;

  void reset(int native) noexcept;
};

struct ServerParallelLoopbackFixture {
  ServerParallelSocketCleanup listener_cleanup{};
  sockaddr_in address{};
  rund::net::Socket listener{};
};

[[nodiscard]] inline bool
ServerCountsAreConsistent(const rund::net::server::Result &result,
                          const std::uint32_t max_accepts) noexcept {
  if (result.started > result.accepted || result.accepted > max_accepts) {
    return false;
  }
  const std::uint64_t admitted =
      static_cast<std::uint64_t>(result.started) + result.rejected;
  const std::uint64_t terminal = static_cast<std::uint64_t>(result.completed) +
                                 result.failed + result.stopped;
  return admitted == result.accepted && terminal == result.started;
}

[[nodiscard]] int
PrepareServerParallelLoopbackListener(ServerParallelLoopbackFixture &fixture);
[[nodiscard]] int
StartServerParallelClientWithByte(const sockaddr_in &address, std::byte byte,
                                  ServerParallelSocketCleanup &cleanup);
[[nodiscard]] int
StartServerParallelSingleClient(const sockaddr_in &address,
                                ServerParallelSocketCleanup &cleanup);
[[nodiscard]] rund::SessionConfig NetServerParallelRunSpec() noexcept;

[[nodiscard]] int RunServerParallelSuccessCase();
[[nodiscard]] int RunServerParallelFrameCapacityCase();
[[nodiscard]] int RunServerParallelHandlerFailureCase();
