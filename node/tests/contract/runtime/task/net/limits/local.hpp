#pragma once

#include "test/assert.hpp"

#include <rund/net/address.hpp>
#include <rund/net/ready/many.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#define NET_LIMIT_ASSERT(condition)                                            \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "NET_LIMIT_ASSERT failed: " #condition << " at "            \
                << __FILE__ << ':' << __LINE__ << '\n';                        \
      return false;                                                            \
    }                                                                          \
  } while (false)

namespace rund::node::test_contract::net_limits {

struct SocketPairCleanup {
  int left = -1;
  int right = -1;

  ~SocketPairCleanup();
  SocketPairCleanup() = default;
  SocketPairCleanup(const SocketPairCleanup &) = delete;
  SocketPairCleanup &operator=(const SocketPairCleanup &) = delete;
};

struct NativeSocketCleanup {
  int fd = -1;

  ~NativeSocketCleanup();
  NativeSocketCleanup() = default;
  explicit NativeSocketCleanup(int native) noexcept;
  NativeSocketCleanup(const NativeSocketCleanup &) = delete;
  NativeSocketCleanup &operator=(const NativeSocketCleanup &) = delete;

  void release() noexcept;
};

struct SocketCloseGuard {
  rund::net::Socket socket{};

  SocketCloseGuard() = default;
  explicit SocketCloseGuard(rund::net::Socket opened) noexcept;
  SocketCloseGuard(const SocketCloseGuard &) = delete;
  SocketCloseGuard &operator=(const SocketCloseGuard &) = delete;

  SocketCloseGuard(SocketCloseGuard &&) noexcept = default;
  SocketCloseGuard &operator=(SocketCloseGuard &&) noexcept = default;

  [[nodiscard]] rund::net::CloseResult close() noexcept;
};

[[nodiscard]] bool MakeSocketPair(SocketPairCleanup &cleanup);
[[nodiscard]] bool MakeLoopbackListener(int *out_fd, sockaddr_in *out_address);
[[nodiscard]] rund::SessionConfig Config() noexcept;
[[nodiscard]] rund::net::Address LoopbackAnyPort();
[[nodiscard]] rund::net::ready::Request
ReadableRequest(rund::net::SocketView socket) noexcept;

} // namespace rund::node::test_contract::net_limits

namespace rund::node::test_contract {

bool NetReadySetCapacityLimitFailsClosed();
bool NetReadySetSetCapacityLimitFailsClosed();
bool NetReadySetMemberCapacityLimitFailsClosed();
bool NetReadySetMemberLimitFailsClosed();
bool NetIovAndDatagramLimitsFailClosed();
bool LimitsReportActiveState();
bool NetSocketRegistryCapacityFailsClosedInActiveRuntime();
bool NetSocketRegistryOpenCapacityFailsClosed();
bool NetSocketRegistryExternalCloseReleasesCapacity();
bool NetSocketRegistryAcceptCapacityClosesAccepted();
bool NetSocketRegistryFdReuseAdmissionFailsClosed();

} // namespace rund::node::test_contract
