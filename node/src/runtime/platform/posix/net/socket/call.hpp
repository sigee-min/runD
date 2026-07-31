#pragma once

#include "../result.hpp"

#include <sys/socket.h>

#include <cerrno>

#if !defined(MSG_DONTWAIT)
#error "runD POSIX networking requires MSG_DONTWAIT"
#endif

namespace rund::node::posix_net {

[[nodiscard]] inline NativeCallResult PrepareSocketSend(const int fd) noexcept {
#if defined(SO_NOSIGPIPE)
  const int enabled = 1;
  errno = 0;
  const int result =
      ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
  if (result < 0 && errno == ENOTSOCK) {
    return PosixNetResult(0, 0);
  }
  return PosixNetResult(result < 0 ? -1 : 0, result < 0 ? errno : 0);
#elif defined(MSG_NOSIGNAL)
  (void)fd;
  return PosixNetResult(0, 0);
#else
  (void)fd;
  return NativeCallResult{
      .value = -1,
      .error = 0,
      .state = NativeCallState::Unsupported,
  };
#endif
}

[[nodiscard]] constexpr int TryRecvFlags() noexcept { return MSG_DONTWAIT; }

[[nodiscard]] constexpr int SignalSafeSendFlags() noexcept {
#if defined(SO_NOSIGPIPE)
  return 0;
#elif defined(MSG_NOSIGNAL)
  return MSG_NOSIGNAL;
#else
  return 0;
#endif
}

[[nodiscard]] constexpr int DirectSendFlags() noexcept {
  return SignalSafeSendFlags();
}

[[nodiscard]] constexpr int TrySendFlags() noexcept {
  return SignalSafeSendFlags() | MSG_DONTWAIT;
}

} // namespace rund::node::posix_net
