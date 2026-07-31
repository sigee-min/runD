#pragma once

#include <rund/session.hpp>

#include <signal.h>

extern volatile sig_atomic_t g_sigpipe_count;

struct ScopedSigpipeHandler {
  bool active = false;
  struct sigaction previous{};

  ScopedSigpipeHandler() noexcept;
  ~ScopedSigpipeHandler();

  ScopedSigpipeHandler(const ScopedSigpipeHandler &) = delete;
  ScopedSigpipeHandler &operator=(const ScopedSigpipeHandler &) = delete;
};

struct BasicSocketCleanup {
  int fd = -1;

  ~BasicSocketCleanup();

  BasicSocketCleanup() = default;
  explicit BasicSocketCleanup(int &native) noexcept;
  BasicSocketCleanup(const BasicSocketCleanup &) = delete;
  BasicSocketCleanup &operator=(const BasicSocketCleanup &) = delete;

  void release() noexcept;
};

[[nodiscard]] int RunNetBasicSyncCase();
[[nodiscard]] int RunNetBasicEventCase();
[[nodiscard]] int RunNetBasicClosedPeerCase();
[[nodiscard]] int RunNetBasicTaskGuardCase();
