#include "../local.hpp"

#include <unistd.h>

ServerSocketCleanup::~ServerSocketCleanup() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

ServerSocketCleanup::ServerSocketCleanup(const int native) noexcept
    : fd(native) {}

void ServerSocketCleanup::reset(const int native) noexcept {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
  fd = native;
}
