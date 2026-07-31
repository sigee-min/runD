#include "../local.hpp"

#include <unistd.h>

MultiClientSocketCleanup::~MultiClientSocketCleanup() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

MultiClientSocketCleanup::MultiClientSocketCleanup(const int native) noexcept
    : fd(native) {}

MultiClientSocketCleanup::MultiClientSocketCleanup(
    MultiClientSocketCleanup&& other) noexcept
    : fd(other.fd) {
  other.fd = -1;
}

MultiClientSocketCleanup& MultiClientSocketCleanup::operator=(
    MultiClientSocketCleanup&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
  fd = other.fd;
  other.fd = -1;
  return *this;
}

void MultiClientSocketCleanup::reset(const int native) noexcept {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
  fd = native;
}
