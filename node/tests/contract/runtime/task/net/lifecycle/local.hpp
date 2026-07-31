#pragma once

#include <rund/session.hpp>

namespace rund::node::test_contract::net_lifecycle {

struct SocketPairCleanup {
  int left = -1;
  int right = -1;

  ~SocketPairCleanup();
  SocketPairCleanup() = default;
  SocketPairCleanup(const SocketPairCleanup &) = delete;
  SocketPairCleanup &operator=(const SocketPairCleanup &) = delete;
};

bool MakeSocketPair(SocketPairCleanup &cleanup);
bool ForceLeftFd(SocketPairCleanup &cleanup, int fd) noexcept;
rund::SessionConfig RunSpec() noexcept;

} // namespace rund::node::test_contract::net_lifecycle

int RunNetLifecycleInvalidCloseCase();
int RunNetLifecycleCloseInvalidatesWaitCase();
int RunNetLifecycleOwnerCase();
int RunNetLifecycleLeaseCase();
int RunNetLifecycleReuseCase();
int RunNetLifecycleTicketCase();
