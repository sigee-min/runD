#include "../io.hpp"
#include "../../reactor/readiness/handle.hpp"
#include "../../reactor/readiness/mask.hpp"
#include "buffer.hpp"
#include "probe.hpp"

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>

namespace rund::node {
namespace {

[[nodiscard]] NativeIoResult
WriteWithoutSigpipe(const int fd,
                    const std::span<const std::byte> buffer) noexcept {
#if defined(F_SETNOSIGPIPE)
  // Restoring this descriptor-local bit would race concurrent writers. The
  // monotonic transition keeps SIGPIPE suppression local to the admitted fd.
  errno = 0;
  if (::fcntl(fd, F_SETNOSIGPIPE, 1) != 0) {
    return NativeIoResult{.err = errno};
  }

  errno = 0;
  const ssize_t value =
      ::write(fd, static_cast<const void *>(buffer.data()), buffer.size());
  return NativeIoResult{.value = static_cast<std::int64_t>(value),
                        .err = value < 0 ? errno : 0};
#else
  sigset_t pipe_mask{};
  if (sigemptyset(&pipe_mask) != 0 || sigaddset(&pipe_mask, SIGPIPE) != 0) {
    return NativeIoResult{.err = errno};
  }

  sigset_t previous_mask{};
  const int mask_error =
      ::pthread_sigmask(SIG_BLOCK, &pipe_mask, &previous_mask);
  if (mask_error != 0) {
    return NativeIoResult{.err = mask_error};
  }
  sigset_t pending{};
  if (::sigpending(&pending) != 0) {
    const int pending_error = errno;
    static_cast<void>(::pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr));
    return NativeIoResult{.err = pending_error};
  }
  const int pending_member = sigismember(&pending, SIGPIPE);
  if (pending_member < 0) {
    const int membership_error = errno;
    static_cast<void>(::pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr));
    return NativeIoResult{.err = membership_error};
  }
  const bool previously_pending = pending_member != 0;

  errno = 0;
  const ssize_t value =
      ::write(fd, static_cast<const void *>(buffer.data()), buffer.size());
  const int write_error = value < 0 ? errno : 0;

  if (value < 0 && write_error == EPIPE && !previously_pending) {
    sigset_t current_pending{};
    if (::sigpending(&current_pending) == 0 &&
        sigismember(&current_pending, SIGPIPE) == 1) {
      int consumed_signal = 0;
      int wait_error = 0;
      do {
        wait_error = ::sigwait(&pipe_mask, &consumed_signal);
      } while (wait_error == EINTR);
    }
  }
  static_cast<void>(::pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr));
  return NativeIoResult{.value = static_cast<std::int64_t>(value),
                        .err = write_error};
#endif
}

} // namespace

ReactorPlatformHandleIdentity
DescribeReactorPlatformHandle(const ReactorHandle handle) noexcept {
  const NativeFdIdentity identity = NativeDescribeFdIdentity(PosixFd(handle));
  return identity.ok
             ? ReactorPlatformHandleIdentity::described(
                   identity.device, identity.inode,
                   static_cast<std::uint32_t>(identity.mode))
             : ReactorPlatformHandleIdentity::invalid();
}

ReactorHandle RetainReactorPlatformHandle(const ReactorHandle handle) noexcept {
  const int fd = PosixFd(handle);
#if defined(F_DUPFD_CLOEXEC)
  const int retained = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
#else
  const int retained = ::dup(fd);
  if (retained < 0) {
    return kInvalidReactorHandle;
  }
  if (::fcntl(retained, F_SETFD, FD_CLOEXEC) != 0) {
    static_cast<void>(::close(retained));
    return kInvalidReactorHandle;
  }
#endif
  return ReactorHandleFromPublic(retained);
}

void ReleaseReactorPlatformHandle(const ReactorHandle handle) noexcept {
  if (handle != kInvalidReactorHandle) {
    static_cast<void>(::close(PosixFd(handle)));
  }
}

int PosixFd(const ReactorHandle handle) noexcept {
  if (handle == kInvalidReactorHandle ||
      handle > static_cast<ReactorHandle>(std::numeric_limits<int>::max())) {
    return -1;
  }
  return static_cast<int>(handle);
}

short PosixInterest(const ReactorInterest interest) noexcept {
  short native = 0;
  if (HasReactorInterest(interest, ReactorInterest::Read)) {
    native = static_cast<short>(native | POLLIN);
  }
  if (HasReactorInterest(interest, ReactorInterest::Write)) {
    native = static_cast<short>(native | POLLOUT);
  }
  return native;
}

ReactorEvent PosixEvents(const short events) noexcept {
  ReactorEvent normalized = ReactorEvent::None;
  if ((events & POLLIN) != 0)
    normalized |= ReactorEvent::Read;
  if ((events & POLLOUT) != 0)
    normalized |= ReactorEvent::Write;
  if ((events & POLLERR) != 0)
    normalized |= ReactorEvent::Error;
  if ((events & POLLHUP) != 0)
    normalized |= ReactorEvent::Hangup;
  return normalized;
}

bool IoPollInvalid(const short revents) noexcept {
  return (revents & POLLNVAL) != 0;
}

bool NativeFdValid(const int fd) noexcept {
  errno = 0;
  return ::fcntl(fd, F_GETFL, 0) >= 0;
}

NativeFdIdentity NativeDescribeFdIdentity(const int fd) noexcept {
  struct stat metadata{};
  errno = 0;
  if (::fstat(fd, &metadata) != 0) {
    return NativeFdIdentity{.err = errno};
  }
  return NativeFdIdentity{
      .ok = true,
      .device = static_cast<std::uint64_t>(metadata.st_dev),
      .inode = static_cast<std::uint64_t>(metadata.st_ino),
      .mode = static_cast<std::uint32_t>(metadata.st_mode),
      .type = static_cast<std::uint32_t>(metadata.st_mode & S_IFMT),
  };
}

bool NativeIsNonblockingFd(const int fd) noexcept {
  errno = 0;
  const int flags = ::fcntl(fd, F_GETFL, 0);
  return flags >= 0 && (flags & O_NONBLOCK) != 0;
}

NativeIoResult NativeSetNonblockingFd(const int fd,
                                      const bool enabled) noexcept {
  errno = 0;
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    return NativeIoResult{.err = errno};
  }
  const int target_flags =
      enabled ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
  if (target_flags == flags) {
    return NativeIoResult{.value = 0};
  }
  errno = 0;
  const int value = ::fcntl(fd, F_SETFL, target_flags);
  return NativeIoResult{.value = static_cast<std::int64_t>(value),
                        .err = value < 0 ? errno : 0};
}

NativeIoResult NativeRead(const int fd,
                          const std::span<std::byte> buffer) noexcept {
  if (!posix::buffer::valid(buffer.data(), buffer.size())) {
    return NativeIoResult{.err = EINVAL, .invalid_buffer = true};
  }
  errno = 0;
  const ssize_t value =
      ::read(fd, static_cast<void *>(buffer.data()), buffer.size());
  return NativeIoResult{.value = static_cast<std::int64_t>(value),
                        .err = value < 0 ? errno : 0};
}

NativeIoResult NativeWrite(const int fd,
                           const std::span<const std::byte> buffer) noexcept {
  if (!posix::buffer::valid(buffer.data(), buffer.size())) {
    return NativeIoResult{.err = EINVAL, .invalid_buffer = true};
  }
  return WriteWithoutSigpipe(fd, buffer);
}

NativeIoResult NativePread(const int fd, const std::span<std::byte> buffer,
                           const std::uint64_t offset) noexcept {
  if (!posix::buffer::valid(buffer.data(), buffer.size())) {
    return NativeIoResult{.err = EINVAL, .invalid_buffer = true};
  }
  constexpr auto kMaxOffset =
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
  if (offset > kMaxOffset) {
    return NativeIoResult{.err = EINVAL};
  }
  errno = 0;
  const ssize_t value = ::pread(fd, static_cast<void *>(buffer.data()),
                                buffer.size(), static_cast<off_t>(offset));
  return NativeIoResult{.value = static_cast<std::int64_t>(value),
                        .err = value < 0 ? errno : 0};
}

NativeIoResult NativeOpen(const std::string_view path, const int flags,
                          const std::uint32_t mode) noexcept {
  std::string copied_path{};
  try {
    copied_path.assign(path);
  } catch (...) {
    return NativeIoResult{.err = ENOMEM};
  }
  errno = 0;
  const int value =
      ::open(copied_path.c_str(), flags, static_cast<mode_t>(mode));
  return NativeIoResult{.value = static_cast<std::int64_t>(value),
                        .err = value < 0 ? errno : 0};
}

NativeIoResult NativeClose(const int fd) noexcept {
  errno = 0;
  const int value = ::close(fd);
  return NativeIoResult{.value = static_cast<std::int64_t>(value),
                        .err = value < 0 ? errno : 0};
}

} // namespace rund::node
