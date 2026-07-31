#include "local.hpp"

#include <sys/socket.h>
#include <unistd.h>

ReadinessSocketCleanup::~ReadinessSocketCleanup() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

ReadinessSocketCleanup::ReadinessSocketCleanup(int &native) noexcept
    : fd(native) {
  native = -1;
}

bool MakeReadinessSocketPair(int (&sockets)[2]) {
  sockets[0] = -1;
  sockets[1] = -1;
  return ::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0;
}
