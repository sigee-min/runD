#include "../local.hpp"

#include <unistd.h>

ServerParallelSocketCleanup::~ServerParallelSocketCleanup() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

ServerParallelSocketCleanup::ServerParallelSocketCleanup(
    const int native) noexcept
    : fd(native) {}

void ServerParallelSocketCleanup::reset(const int native) noexcept {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
  fd = native;
}
