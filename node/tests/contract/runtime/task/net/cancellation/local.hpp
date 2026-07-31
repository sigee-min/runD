#pragma once

#include <rund/session.hpp>

struct CancelSocketPairCleanup {
  int left = -1;
  int right = -1;

  ~CancelSocketPairCleanup();

  CancelSocketPairCleanup() = default;
  CancelSocketPairCleanup(const CancelSocketPairCleanup &) = delete;
  CancelSocketPairCleanup &operator=(const CancelSocketPairCleanup &) = delete;
};

[[nodiscard]] bool MakeCancelSocketPair(CancelSocketPairCleanup &cleanup);
[[nodiscard]] bool SaturateSocket(int fd);
[[nodiscard]] rund::SessionConfig NetCancellationRunSpec() noexcept;

[[nodiscard]] int RunNetCancellationReadableWakeCase();
[[nodiscard]] int RunNetCancellationCleanupCase();
[[nodiscard]] int RunNetCancellationWritableCase();
[[nodiscard]] int RunNetCancellationReadyManyCase();
