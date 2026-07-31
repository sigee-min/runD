#include "local.hpp"

#include <sys/socket.h>
#include <unistd.h>

volatile sig_atomic_t g_sigpipe_count = 0;

namespace {

void CountSigpipe(const int signal) noexcept {
  if (signal == SIGPIPE) {
    g_sigpipe_count = static_cast<sig_atomic_t>(g_sigpipe_count + 1);
  }
}

}  // namespace

ScopedSigpipeHandler::ScopedSigpipeHandler() noexcept {
  struct sigaction action {};
  action.sa_handler = CountSigpipe;
  static_cast<void>(sigemptyset(&action.sa_mask));
  active = ::sigaction(SIGPIPE, &action, &previous) == 0;
}

ScopedSigpipeHandler::~ScopedSigpipeHandler() {
  if (active) {
    static_cast<void>(::sigaction(SIGPIPE, &previous, nullptr));
  }
}

BasicSocketCleanup::~BasicSocketCleanup() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

BasicSocketCleanup::BasicSocketCleanup(int &native) noexcept : fd(native) {
  native = -1;
}

void BasicSocketCleanup::release() noexcept { fd = -1; }
