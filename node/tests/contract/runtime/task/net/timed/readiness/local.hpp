#pragma once

#include <rund/session.hpp>

#include <cstdint>
#include <memory>

struct TimedSocketPairCleanup {
  int left = -1;
  int right = -1;

  ~TimedSocketPairCleanup();

  TimedSocketPairCleanup() = default;
  TimedSocketPairCleanup(const TimedSocketPairCleanup &) = delete;
  TimedSocketPairCleanup &operator=(const TimedSocketPairCleanup &) = delete;
};

[[nodiscard]] bool MakeTimedSocketPair(TimedSocketPairCleanup &cleanup);
[[nodiscard]] rund::SessionConfig NetTimedReadinessRunSpec() noexcept;

[[nodiscard]] int RunTimedReadinessReadyCase();
[[nodiscard]] int RunTimedReadinessCloseCase();
[[nodiscard]] int RunTimedReadinessInvalidCase();
[[nodiscard]] int RunTimedReadinessCoroutineCase();
