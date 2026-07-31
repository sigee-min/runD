#pragma once

#include <rund/session.hpp>

struct ReadinessSocketCleanup {
  int fd = -1;

  ~ReadinessSocketCleanup();

  explicit ReadinessSocketCleanup(int &native) noexcept;
  ReadinessSocketCleanup(const ReadinessSocketCleanup &) = delete;
  ReadinessSocketCleanup &operator=(const ReadinessSocketCleanup &) = delete;
};

[[nodiscard]] bool MakeReadinessSocketPair(int (&sockets)[2]);

[[nodiscard]] int RunNetReadinessBasicCase();
[[nodiscard]] int RunNetReadinessInvalidCase();
[[nodiscard]] int RunNetReadinessWritableCase();
[[nodiscard]] int RunNetReadinessParkedCase();
[[nodiscard]] int RunNetReadinessHupCase();
[[nodiscard]] int RunNetReadinessZeroCase();
