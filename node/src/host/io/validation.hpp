#pragma once

#include "access.hpp"

namespace rund::host::io::detail {

struct FdIdentity final {
  int native = -1;
  std::uint64_t host_id = 0u;

  [[nodiscard]] constexpr bool live() const noexcept {
    return native >= 0 && host_id == static_cast<std::uint64_t>(native) + 1u;
  }

  [[nodiscard]] constexpr bool replay() const noexcept {
    return native == -1 && host_id != 0u;
  }
};

[[nodiscard]] constexpr FdIdentity Project(const FdView fd) noexcept {
  return FdIdentity{.native = Access::native(fd), .host_id = Access::id(fd)};
}

} // namespace rund::host::io::detail
