#pragma once

#include "test/assert.hpp"

#include <rund/net/address.hpp>
#include <rund/net/listener.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#define DATAGRAM_ASSERT(condition)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "DATAGRAM_ASSERT failed: " #condition << " at " << __FILE__ \
                << ':' << __LINE__ << '\n';                                    \
      return false;                                                            \
    }                                                                          \
  } while (false)

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

struct RawFdGuard {
  int fd = -1;

  RawFdGuard() = default;
  explicit RawFdGuard(int native) noexcept;
  RawFdGuard(const RawFdGuard &) = delete;
  RawFdGuard &operator=(const RawFdGuard &) = delete;
  ~RawFdGuard();
};

[[nodiscard]] rund::net::Address LoopbackAnyPort();
[[nodiscard]] bool IsLoopbackPort(rund::net::Address address) noexcept;
[[nodiscard]] bool OpenBoundUdpPair(SocketCloseGuard &sender,
                                    SocketCloseGuard &receiver,
                                    rund::net::LocalResult *sender_address,
                                    rund::net::LocalResult *receiver_address);

bool NetDatagramSendReceiveLoopback();
bool NetDatagramWouldBlockAndInvalidInputsFailClosed();
bool NetDatagramPreflightFailuresFailClosed();
bool NetDatagramTryUsesPerCallNonblocking();
bool NetDatagramRejectsRuntimeCapacityOversize();
bool NetDatagramReplayEventsAreStable();
bool NetDatagramRejectsOversizedRequest();
bool NetDatagramTransfersEmptyPacket();
