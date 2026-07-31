#include "../local.hpp"

#include "test/assert.hpp"

#include <poll.h>
#include <sys/socket.h>

namespace {

[[nodiscard]] bool SendAll(const int fd, const std::byte byte) {
  const unsigned char raw = static_cast<unsigned char>(byte);
  const char* cursor = reinterpret_cast<const char*>(&raw);
  std::size_t remaining = 1u;
  while (remaining > 0u) {
    const ssize_t sent = ::send(fd, cursor, remaining, 0);
    if (sent <= 0) {
      return false;
    }
    cursor += sent;
    remaining -= static_cast<std::size_t>(sent);
  }
  return true;
}

}  // namespace

bool RecvMultiClientOneWithTimeout(const int fd, std::byte& out) {
  pollfd poll_fd{.fd = fd, .events = POLLIN, .revents = 0};
  if (::poll(&poll_fd, 1u, 1000) != 1) {
    return false;
  }
  unsigned char raw = 0u;
  if (::recv(fd, &raw, 1u, 0) != 1) {
    return false;
  }
  out = static_cast<std::byte>(raw);
  return true;
}

int StartMultiClientBlockingClient(const sockaddr_in& connect_address,
                                   const std::byte byte,
                                   MultiClientSocketCleanup& client_cleanup) {
  const int client_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  TEST_ASSERT(client_fd >= 0);
  client_cleanup.reset(client_fd);
  TEST_ASSERT(::connect(client_fd,
                        reinterpret_cast<const sockaddr*>(&connect_address),
                        sizeof(connect_address)) == 0);
  TEST_ASSERT(SendAll(client_fd, byte));
  return 0;
}
