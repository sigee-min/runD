#pragma once

#include "../readiness/state.hpp"

#include <cstdint>

namespace rund::node {

enum class ReactorPlatformHandleIdentityDisposition : std::uint8_t {
  Invalid,
  Described,
};

class ReactorPlatformHandleIdentity final {
public:
  [[nodiscard]] static constexpr ReactorPlatformHandleIdentity
  invalid() noexcept {
    return ReactorPlatformHandleIdentity{
        ReactorPlatformHandleIdentityDisposition::Invalid, 0u, 0u, 0u};
  }

  [[nodiscard]] static constexpr ReactorPlatformHandleIdentity
  described(const std::uint64_t device, const std::uint64_t inode,
            const std::uint32_t mode) noexcept {
    return ReactorPlatformHandleIdentity{
        ReactorPlatformHandleIdentityDisposition::Described, device, inode,
        mode};
  }

  [[nodiscard]] constexpr ReactorPlatformHandleIdentityDisposition
  disposition() const noexcept {
    return disposition_;
  }
  [[nodiscard]] constexpr std::uint64_t device() const noexcept {
    return device_;
  }
  [[nodiscard]] constexpr std::uint64_t inode() const noexcept {
    return inode_;
  }
  [[nodiscard]] constexpr std::uint32_t mode() const noexcept { return mode_; }

  [[nodiscard]] constexpr bool
  same_object(const ReactorPlatformHandleIdentity &other) const noexcept {
    return disposition_ ==
               ReactorPlatformHandleIdentityDisposition::Described &&
           other.disposition_ ==
               ReactorPlatformHandleIdentityDisposition::Described &&
           device_ == other.device_ && inode_ == other.inode_ &&
           mode_ == other.mode_;
  }

private:
  constexpr ReactorPlatformHandleIdentity(
      const ReactorPlatformHandleIdentityDisposition disposition,
      const std::uint64_t device, const std::uint64_t inode,
      const std::uint32_t mode) noexcept
      : device_(device), inode_(inode), mode_(mode), disposition_(disposition) {
  }

  std::uint64_t device_;
  std::uint64_t inode_;
  std::uint32_t mode_;
  ReactorPlatformHandleIdentityDisposition disposition_;
};

[[nodiscard]] ReactorPlatformHandleIdentity
    DescribeReactorPlatformHandle(ReactorHandle) noexcept;
[[nodiscard]] ReactorHandle RetainReactorPlatformHandle(ReactorHandle) noexcept;
void ReleaseReactorPlatformHandle(ReactorHandle) noexcept;

} // namespace rund::node
