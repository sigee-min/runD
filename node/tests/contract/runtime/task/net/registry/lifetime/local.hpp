#pragma once

namespace rund::node::test_contract::net_registry_lifetime {

struct SocketPair {
  int left = -1;
  int right = -1;

  ~SocketPair();
  SocketPair() = default;
  SocketPair(const SocketPair &) = delete;
  SocketPair &operator=(const SocketPair &) = delete;

  static void Close(int &fd) noexcept;
};

bool MakeSocketPair(SocketPair &pair) noexcept;
bool ForceLeftFd(SocketPair &pair, int fd) noexcept;

} // namespace rund::node::test_contract::net_registry_lifetime

int RunNetRegistryLifetimeGenerationCase();
int RunNetRegistryLifetimeReplacementCase();
int RunNetRegistryLifetimeStabilityCase();
