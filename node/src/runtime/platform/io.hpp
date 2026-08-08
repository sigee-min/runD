#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string_view>

namespace rund::node {

enum class NativeIoDisposition : std::uint8_t {
  Complete,
  Failed,
  InvalidBuffer,
  Unsupported,
};

class NativeIoResult final {
public:
  NativeIoResult() = delete;

  [[nodiscard]] static constexpr NativeIoResult
  complete(const std::int64_t value) noexcept {
    if (value < 0) {
      std::abort();
    }
    return NativeIoResult{NativeIoDisposition::Complete, value, 0};
  }

  [[nodiscard]] static constexpr NativeIoResult
  invalid_buffer(const int native_error) noexcept {
    if (native_error == 0) {
      std::abort();
    }
    return NativeIoResult{NativeIoDisposition::InvalidBuffer, -1, native_error};
  }

  [[nodiscard]] static constexpr NativeIoResult
  failed(const int native_error) noexcept {
    if (native_error == 0) {
      std::abort();
    }
    return NativeIoResult{NativeIoDisposition::Failed, -1, native_error};
  }

  [[nodiscard]] static constexpr NativeIoResult unsupported() noexcept {
    return NativeIoResult{NativeIoDisposition::Unsupported, -1, 0};
  }

  [[nodiscard]] constexpr NativeIoDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr std::int64_t value() const noexcept { return value_; }

  [[nodiscard]] constexpr int native_error() const noexcept {
    return native_error_;
  }

private:
  constexpr NativeIoResult(const NativeIoDisposition disposition,
                           const std::int64_t value,
                           const int native_error) noexcept
      : value_(value), native_error_(native_error), disposition_(disposition) {}

  std::int64_t value_;
  int native_error_;
  NativeIoDisposition disposition_;
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
