#include "../local.hpp"

#include "test/assert.hpp"

#include <sys/socket.h>

namespace {

[[nodiscard]] bool SendAll(const int fd, const std::byte byte) {
  const unsigned char raw = static_cast<unsigned char>(byte);
  const char *cursor = reinterpret_cast<const char *>(&raw);
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

} // namespace

int StartServerParallelClientWithByte(const sockaddr_in &address,
                                      const std::byte byte,
                                      ServerParallelSocketCleanup &cleanup) {
  const int client_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  TEST_ASSERT(client_fd >= 0);
  cleanup.reset(client_fd);
  TEST_ASSERT(::connect(client_fd, reinterpret_cast<const sockaddr *>(&address),
                        sizeof(address)) == 0);
  TEST_ASSERT(SendAll(cleanup.fd, byte));
  return 0;
}

int StartServerParallelSingleClient(const sockaddr_in &address,
                                    ServerParallelSocketCleanup &cleanup) {
  return StartServerParallelClientWithByte(address, std::byte{0x7Fu}, cleanup);
}
