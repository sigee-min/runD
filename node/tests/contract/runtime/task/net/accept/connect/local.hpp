#pragma once

#include <rund/net/address.hpp>
#include <rund/session.hpp>

#include <netinet/in.h>

struct AcceptConnectSocketCleanup {
  int fd = -1;

  ~AcceptConnectSocketCleanup();

  AcceptConnectSocketCleanup() = default;
  explicit AcceptConnectSocketCleanup(int native) noexcept;
  AcceptConnectSocketCleanup(const AcceptConnectSocketCleanup &) = delete;
  AcceptConnectSocketCleanup &
  operator=(const AcceptConnectSocketCleanup &) = delete;

  void reset(int native) noexcept;
  void release() noexcept;
};

[[nodiscard]] rund::net::Address
AcceptConnectAddressFromSockaddr(const sockaddr_in &address);
[[nodiscard]] bool MakeAcceptConnectLoopbackListener(int *out_fd,
                                                     sockaddr_in *out_address);

[[nodiscard]] int RunAcceptConnectBasicCase();
[[nodiscard]] int RunAcceptConnectEventCase();
[[nodiscard]] int RunAcceptConnectRefusedCase();
