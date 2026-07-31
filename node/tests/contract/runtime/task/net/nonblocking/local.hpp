#pragma once

#include <rund/session.hpp>

struct NonblockingSocketCleanup {
  int fd = -1;

  ~NonblockingSocketCleanup();

  explicit NonblockingSocketCleanup(int &native) noexcept;
  NonblockingSocketCleanup(const NonblockingSocketCleanup &) = delete;
  NonblockingSocketCleanup &
  operator=(const NonblockingSocketCleanup &) = delete;
};

[[nodiscard]] bool MakeSocketPair(int (&sockets)[2]);
[[nodiscard]] rund::SessionConfig
NetNonblockingRunSpec(std::uint32_t host_event_capacity = 4u) noexcept;

[[nodiscard]] int RunNetNonblockingBasicCase();
[[nodiscard]] int RunNetTryPerCallNonblockingCase();
[[nodiscard]] int RunNetNonblockingPartialAndNullCase();
[[nodiscard]] int RunNetNonblockingPressureCase();
[[nodiscard]] int RunNetNonblockingReplayCase();
