#include "../io.hpp"
#include "../../reactor/platform.hpp"

namespace rund::node {
namespace {

[[nodiscard]] constexpr NativeIoResult UnsupportedIo() noexcept {
  return NativeIoResult::unsupported();
}

} // namespace

bool NativeFdValid(const int fd) noexcept {
  (void)fd;
  return false;
}

NativeFdIdentity NativeDescribeFdIdentity(const int fd) noexcept {
  (void)fd;
  return NativeFdIdentity::invalid();
}

bool NativeIsNonblockingFd(const int fd) noexcept {
  (void)fd;
  return false;
}

NativeIoResult NativeSetNonblockingFd(const int fd,
                                      const bool enabled) noexcept {
  (void)fd;
  (void)enabled;
  return UnsupportedIo();
}

NativeIoResult NativeRead(const int fd,
                          const std::span<std::byte> buffer) noexcept {
  (void)fd;
  (void)buffer;
  return UnsupportedIo();
}

NativeIoResult NativeWrite(const int fd,
                           const std::span<const std::byte> buffer) noexcept {
  (void)fd;
  (void)buffer;
  return UnsupportedIo();
}

NativeIoResult NativePread(const int fd, const std::span<std::byte> buffer,
                           const std::uint64_t offset) noexcept {
  (void)fd;
  (void)buffer;
  (void)offset;
  return UnsupportedIo();
}

NativeIoResult NativeOpen(const std::string_view path, const int flags,
                          const std::uint32_t mode) noexcept {
  (void)path;
  (void)flags;
  (void)mode;
  return UnsupportedIo();
}

NativeIoResult NativeClose(const int fd) noexcept {
  (void)fd;
  return UnsupportedIo();
}

ReactorPlatformHandleIdentity
DescribeReactorPlatformHandle(const ReactorHandle handle) noexcept {
  (void)handle;
  return ReactorPlatformHandleIdentity::invalid();
}

ReactorHandle RetainReactorPlatformHandle(const ReactorHandle handle) noexcept {
  (void)handle;
  return kInvalidReactorHandle;
}

void ReleaseReactorPlatformHandle(const ReactorHandle handle) noexcept {
  (void)handle;
}

} // namespace rund::node
