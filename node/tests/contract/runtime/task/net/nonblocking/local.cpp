#include "local.hpp"

#include <sys/socket.h>
#include <unistd.h>

NonblockingSocketCleanup::~NonblockingSocketCleanup() {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
  }
}

NonblockingSocketCleanup::NonblockingSocketCleanup(int &native) noexcept
    : fd(native) {
  native = -1;
}

bool MakeSocketPair(int (&sockets)[2]) {
  sockets[0] = -1;
  sockets[1] = -1;
  return ::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0;
}

rund::SessionConfig
NetNonblockingRunSpec(const std::uint32_t host_event_capacity) noexcept {
  return rund::SessionConfig{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 2u,
              .ready_queue_capacity = 4u,
              .host_event_capacity = host_event_capacity,
          },
  };
}
