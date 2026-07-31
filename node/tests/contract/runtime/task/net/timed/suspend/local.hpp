#pragma once

#include <rund/session.hpp>

#include <cstdint>
#include <memory>

struct TimedSuspendSocketPairCleanup {
  int left = -1;
  int right = -1;

  ~TimedSuspendSocketPairCleanup();

  TimedSuspendSocketPairCleanup() = default;
  TimedSuspendSocketPairCleanup(const TimedSuspendSocketPairCleanup &) = delete;
  TimedSuspendSocketPairCleanup &
  operator=(const TimedSuspendSocketPairCleanup &) = delete;
};

[[nodiscard]] bool
MakeTimedSuspendSocketPair(TimedSuspendSocketPairCleanup &cleanup);
[[nodiscard]] rund::SessionConfig TimedReadySuspendRunSpec() noexcept;

[[nodiscard]] int RunTimedSuspendCoroutineCase();
