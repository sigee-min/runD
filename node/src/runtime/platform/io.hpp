#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rund::node {

struct NativeIoResult {
  std::int64_t value = -1;
  int err = 0;
  bool invalid_buffer = false;
  bool unsupported = false;
};

struct NativeFdIdentity {
  bool ok = false;
  int err = 0;
  std::uint64_t device = 0u;
  std::uint64_t inode = 0u;
  std::uint32_t mode = 0u;
  std::uint32_t type = 0u;
};

[[nodiscard]] bool NativeFdValid(int fd) noexcept;
[[nodiscard]] NativeFdIdentity NativeDescribeFdIdentity(int fd) noexcept;
[[nodiscard]] bool NativeIsNonblockingFd(int fd) noexcept;
[[nodiscard]] NativeIoResult NativeSetNonblockingFd(int fd,
                                                    bool enabled) noexcept;
[[nodiscard]] NativeIoResult NativeRead(int fd,
                                        std::span<std::byte> buffer) noexcept;
[[nodiscard]] NativeIoResult
NativeWrite(int fd, std::span<const std::byte> buffer) noexcept;
[[nodiscard]] NativeIoResult NativePread(int fd, std::span<std::byte> buffer,
                                         std::uint64_t offset) noexcept;
[[nodiscard]] NativeIoResult NativeOpen(std::string_view path, int flags,
                                        std::uint32_t mode) noexcept;
[[nodiscard]] NativeIoResult NativeClose(int fd) noexcept;

} // namespace rund::node
