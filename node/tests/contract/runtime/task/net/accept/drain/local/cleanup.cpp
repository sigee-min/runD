#include "../local.hpp"

#include <unistd.h>
#include <utility>

AcceptDrainSocketCleanup::~AcceptDrainSocketCleanup() {
  if (!socket && fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

AcceptDrainSocketCleanup::AcceptDrainSocketCleanup(const int native) noexcept
    : fd(native) {}

AcceptDrainSocketCleanup::AcceptDrainSocketCleanup(
    AcceptDrainSocketCleanup &&other) noexcept
    : socket(std::move(other.socket)), fd(other.fd) {
  other.fd = -1;
}

AcceptDrainSocketCleanup &
AcceptDrainSocketCleanup::operator=(AcceptDrainSocketCleanup &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (!socket && fd >= 0) {
    static_cast<void>(::close(fd));
  }
  socket = std::move(other.socket);
  fd = other.fd;
  other.fd = -1;
  return *this;
}
